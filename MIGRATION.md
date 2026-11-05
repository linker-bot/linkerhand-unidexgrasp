# Official L20 right-hand model: state PPO migration

This implementation targets the state-based PPO task. It uses the right-hand
model from [`linkerhand-urdf`](https://github.com/linker-bot/linkerhand-urdf/tree/49f26021b0a91ff5c33150df22970c07bb548672/L20/right)
at commit `49f26021b0a91ff5c33150df22970c07bb548672`.
The upstream repository is a submodule at `third_party/linkerhand-urdf`, pinned
to that commit by the parent repository. Source and derived asset locations are
separate:

```text
third_party/linkerhand-urdf/L20/right/  # upstream URDF and visual meshes
l20/assets/l20_right/                 # runtime URDF, collision meshes, manifests
```

The source repository and asset checksums are recorded in `SOURCE.json` and
`l20/assets/l20_right/manifest.json`. Initialization is explicit; model loading
and training do not download dependencies or update the submodule.

Local validation covers kinematics, data conversion, CPU tensors, and mocked
simulator interfaces. The implementation has not been executed in Isaac Gym,
used for a complete PPO training run, or tested on hardware. The vision and
DAgger tasks retain their existing ShadowHand interfaces.

## Joint model and retargeting

The official URDF contains 21 revolute joints and five mimic relationships,
leaving 16 independent joint variables under the specified constraints. This
count describes the URDF; it does not establish the number of hardware motors.

`joint-map.json` records the anatomical correspondence with the legacy model:

```text
Shadow J3 -> legacy joint0 -> official mcp_roll
Shadow J2 -> legacy joint1 -> official mcp_pitch
Shadow J1 -> legacy joint2 -> official pip
Shadow J0 -> legacy joint3 -> official dip

FF -> index_finger  -> index
MF -> middle_finger -> middle
RF -> ring_finger   -> ring
LF -> little_finger -> pinky

THJ4 -> thumb_joint0 -> thumb_cmc_yaw
THJ3 -> thumb_joint1 -> thumb_cmc_roll
THJ2 -> thumb_joint2 -> thumb_cmc_pitch
THJ1 -> thumb_joint3 -> thumb_mcp
THJ0 -> thumb_joint4 -> thumb_dip
```

These correspondences do not define a scalar angle conversion. Joint axes,
origins, limits, and thumb frames differ between the models. `l20/shadow.py`
computes five fingertip and five finger-base landmarks from the generator's
MJCF. `l20/retarget.py` fits the target root pose and 16 independent joints to
those landmarks. The effect of ShadowHand's `LFJ4`, which has no separate
target joint, remains in the source forward kinematics.

The previous wrist rotation and `[0, -0.006, -0.063]` metre translation are
initial guesses only. Retargeting optimizes both root rotation and translation.

The mimic relationships are read from the official URDF:

```text
index_dip  = 0.8917 * index_pip
middle_dip = 0.8917 * middle_pip
ring_dip   = 0.8917 * ring_pip
pinky_dip  = 0.8917 * pinky_pip
thumb_dip  = 1.1619 * thumb_mcp
```

Independent limits include the corresponding child-joint limits. Prior data,
position targets, reset positions, and reset velocities use the same expansion.

## Runtime assets and control

The source URDF and its 22 STL files are read from the submodule.
`python -m l20.prepare` verifies the checked-out commit and source-file
checksums, then prepares a separate runtime URDF in this repository:

- Mimic tags are removed so the importer exposes all 21 physical joints.
- The controller expands 16 independent targets into those joints by name.
- Collision geometry uses one convex hull per link. Visual geometry, mass, and
  inertia remain from the source URDF.

Runtime visual paths point to the submodule using relative paths. Generated
URDFs, convex meshes, and manifests stay under `l20/assets/l20_right`; preparation
does not write to the upstream checkout. Derived collision assets remain tracked
because they participate in prior and checkpoint compatibility checks.

The source `base_link.STL` is not watertight when loaded with Trimesh. Prepared
convex meshes allow interior queries and are shared by geometric screening and
the simulator. They fill concavities and can exclude otherwise feasible grasps.
Changing to a different collision representation requires updating screening,
runtime assets, priors, and checkpoint metadata together.

Mimic execution is a position-servo approximation, not a rigid transmission
constraint. The code does not overwrite physical joint states after simulation.
It records `mimic_error_rad` and `mimic_error_max_rad`; contact-dependent tracking
error still requires simulator testing. A backend with physical coupling
constraints is needed if exact transmission dynamics are required.

The policy has 22 actions: six root controls followed by these 16 joints:

```text
index_mcp_roll,  index_mcp_pitch,  index_pip
middle_mcp_roll, middle_mcp_pitch, middle_pip
ring_mcp_roll,   ring_mcp_pitch,   ring_pip
pinky_mcp_roll,  pinky_mcp_pitch,  pinky_pip
thumb_cmc_yaw, thumb_cmc_roll, thumb_cmc_pitch, thumb_mcp
```

The 238-element observation contains joint positions and velocities, five
fingertip states, the root pose, all 22 actions, object state, prior-pose error,
16 independent joint errors, and 64 visual features. Joint state is reordered
into a fixed named layout after import. The actor and critic use their supplied
observation and state dimensions.

Pose priors and pose errors refer to `base_link`. Reach/contact reward shaping
uses the official base-link centre of mass as a separate palm reference. Root
forces are applied to the named base body. Reset places each object at its
recorded resting height and starts the hand 0.12 m above the target grasp.

Position-drive stiffness `5.0` and damping `0.1` are initial simulation settings.
They have not been calibrated against the hardware. Legacy PPO weights are
incompatible with the new dimensions and model metadata and are rejected.

## Prior data and coordinate frames

Regenerate priors from ShadowHand generation results. Renaming legacy `arr_0`
fields does not account for the changed kinematics or coupling constraints.

Both generation datasets now retain `object_pose`, the scaled object-to-scene
transform. Export converts hand poses and table planes from scene coordinates
to object coordinates. For old results without this field, recovery requires
an exact match against the first 3,000 source object points in `pcs_table.npy`
and the associated `poses.npy`. Missing or ambiguous matches are rejected.

The NPZ format contains named physical and independent joint arrays, hand
translation and rotation, scale, plane parameters, and JSON metadata. Units are
metres and radians. Metadata includes the URDF revision and checksum, collision
asset checksum, source index and object pose, landmark error, and screening
result. Loading uses `allow_pickle=False` and does not depend on dictionary
insertion order.

Export merges all input batches and assigns a distinct filename to each source
sample. Existing prior files are not overwritten. Rejected samples are recorded
in `export-report.json`.

After kinematic fitting, bounded local refinement can adjust the root pose and
joints to reduce penetration and approach object contact. Final screening uses
the prepared meshes, with these defaults:

- Sampled penetration of at most 2 mm against the object, table, and checked
  non-neighbouring hand links.
- At least two distal links within 15 mm of the object surface.
- A virtual-tip error bound of at most 30 mm.

These are sampled geometric criteria. They do not establish force closure,
continuous collision clearance, or dynamic grasp success. Optimization may
reach its evaluation limit; acceptance depends on the final screening criteria.
`--kinematics-only` exports unscreened candidates, which the policy rejects by
default. `allowUnvalidatedPriors` is intended for explicit interface diagnostics.

Two priors regenerated from the four bundled candidates are included under
`dexgrasp_policy_l20hand/assets_tiny_l20/dataset_l20_right_v2`: bowl and jar.
Bottle and cellphone candidates exceeded the allowed tip displacement after
refinement. The default state configuration enables jar. The two accepted
candidates are not evidence of a 50% dynamic grasp success rate.

## Run and validate

From the repository root, in the existing Python 3.8 environment:

```bash
git submodule update --init --recursive -- third_party/linkerhand-urdf
pip install -r requirements-l20.txt
python -m l20.prepare
python -m pytest -q
```

Use the explicit submodule path: the original repository also contains a CSDF
gitlink without a `.gitmodules` entry, which prevents an unqualified recursive
submodule update. The CPU test workflow initializes only the model dependency.
It runs the tests with Python 3.8 and PyTorch 1.13.1 on Linux; it does not run
Isaac Gym.

When updating the parent repository, run the same scoped submodule update to
check out its recorded model revision. Do not use `git submodule update --remote`
for training. An upstream model update should explicitly change the submodule
commit, source manifest, and model revision constant, followed by asset
preparation, prior regeneration where required, and validation.

Asset preparation verifies and reuses the checked-in convex meshes by default.
`python -m l20.prepare --rebuild` regenerates them. If their checksums change,
regenerate priors and retrain policies rather than reusing old metadata.

To regenerate the bundled results, write to a new directory:

```bash
python dexgrasp_generation_forlinker/tests/data_for_RL.py \
  --input dexgrasp_generation_forlinker/eval/result.pt \
  --mesh-root dexgrasp_generation_forlinker/mini_generation/DFCData/meshes \
  --object-prefix core \
  --output dexgrasp_policy_l20hand/assets_tiny_l20/dataset_l20_right_v2-regenerated
```

Omit `--object-prefix core` when the source object codes already include that
namespace. Set `priorDataset` to the new directory and include only objects
with accepted priors and the required object assets and visual features in
`object_code_dict`.

To inspect a saved prior, use its source index with `--num`:

```bash
python dexgrasp_generation_forlinker/tests/visualize_result_l20_shadow.py \
  --exp_dir dexgrasp_generation_forlinker/eval --num 3 \
  --mesh-root dexgrasp_generation_forlinker/mini_generation/DFCData/meshes \
  --prior dexgrasp_policy_l20hand/assets_tiny_l20/dataset_l20_right_v2/core/jar-7f355d9510fbe2b02af07f81a2ca6736/00000003.npz \
  --show-shadow
```

In the Linux/Isaac Gym environment, run both control modes before training:

```bash
python tools/smoke_l20.py --steps 100 --output smoke-absolute.json
python tools/smoke_l20.py --steps 100 --relative --output smoke-relative.json
```

The smoke script checks loading, reset, finite observations, and measured
coupling error during an open/close cycle. It does not evaluate grasp success.

Then run a short state-PPO integration check using a new log directory:

```bash
cd dexgrasp_policy_l20hand/dexgrasp
python train.py --task ShadowHandGrasp --algo ppo --seed 7 \
  --num_envs 4 --max_iterations 1 --headless \
  --sim_device cuda:0 --rl_device cuda:0 --logdir logs/l20-official-smoke
```

Before increasing training scale, evaluate contact behaviour, coupling error,
lift retention, and success rates on fixed objects, scales, and random seeds.

## Files

`l20/` supplies the shared model, source MJCF kinematics, frame conversion,
retargeting, geometric screening, export, and checkpoint metadata. The
generation dataset and L20 builder use those functions, as do the export and
preview entry points. The state task, PPO network/checkpoint code, task dispatch,
and state configuration use the new dimensions and named joints.

Related fixes remove invalid fingertip indices, the preview/export sign
disagreement, batch loss and filename overwrites, constant force observations,
action truncation, and the unused fixed-table options. The PPO distribution now
passes standard deviation, rather than variance, to `scale_tril`.
