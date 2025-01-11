import jax
import ml_collections
import orbax.checkpoint as ocp
from flax.training import orbax_utils


def create_checkpoint_manager(
    workdir: str,
    config: ml_collections.ConfigDict,
):
    checkpoint_manager_options = ocp.CheckpointManagerOptions(
        create=True, max_to_keep=config.max_to_keep_checkpoint
    )
    checkpoint_manager = ocp.CheckpointManager(
        workdir,
        ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        checkpoint_manager_options,
    )
    return checkpoint_manager


def restore_checkpoint(checkpoint_manager, state):
    restore_args = orbax_utils.restore_args_from_target(state, mesh=None)
    if checkpoint_manager.latest_step() is not None:
        return checkpoint_manager.restore(
            checkpoint_manager.latest_step(),
            items=state,
            restore_kwargs={"restore_args": restore_args},
        )
    else:
        return state


def save_checkpoint(checkpoint_manager, state):
    if jax.process_index() == 0:
        # get train state from the first replica
        state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
        save_args = orbax_utils.save_args_from_target(state)
        step = int(state.step)
        checkpoint_manager.save(step, state, save_kwargs={"save_args": save_args})
