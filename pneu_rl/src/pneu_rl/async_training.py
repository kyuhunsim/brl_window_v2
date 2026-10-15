"""Process-isolated SAC learning. No hardware object enters the learner.

The bounded experience queue never blocks the actor: saturation is an explicit
error. Policy snapshots use a nonblocking lock; a busy learner means the actor
keeps its current policy. This is soft real-time, not a scheduling guarantee.
"""
from __future__ import annotations

import copy
import csv
import os
import multiprocessing as mp
import pickle
import queue
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch


def _policy_array(policy):
    return np.concatenate([v.detach().cpu().numpy().reshape(-1)
                           for v in policy.state_dict().values()]).astype(np.float32)


def _install_policy(policy, values):
    offset = 0
    with torch.no_grad():
        for value in policy.state_dict().values():
            count = value.numel()
            value.copy_(torch.from_numpy(values[offset:offset + count].reshape(value.shape)))
            offset += count


def _publish(policy, shared, lock, version):
    values = _policy_array(policy)
    if lock.acquire(False):
        try:
            np.frombuffer(shared, dtype=np.float32)[:] = values
            version.value += 1
        finally:
            lock.release()


def _learner(payload, experience, status, shared, lock, version, sync_every):
    try:
        torch.set_num_threads(1)
        model = pickle.loads(payload)
        updates = 0
        losses = np.zeros(2)
        episode_updates = 0
        infos, diagnostics = [], []
        status.send(('ready', None))
        while True:
            message = experience.get()
            if message is None:
                break
            kind, data = message
            if kind == 'transition':
                transition, step, can_update, info, diag = data
                model.buffer.add(*transition)
                model.total_steps = step
                if info is not None:
                    infos.append(info)
                    diagnostics.append(diag)
                if (can_update and len(model.buffer) > model.batch_size
                        and step % model.update_every == 0):
                    for _ in range(model.epoch):
                        model.update_parameters()
                        updates += 1
                        episode_updates += 1
                        losses += [model.current_critics_loss, model.current_policy_loss]
                    if updates % sync_every == 0:
                        _publish(model.policy, shared, lock, version)
                model.total_steps = step + 1
            elif kind == 'episode':
                epi, reward, step = data
                if model.log:
                    model.save_model(model.logger.model_path)
                    if model.checkpoint_interval and epi % model.checkpoint_interval == 0:
                        model.save_model(f'{model.logger.folder_path}/model_epi_{epi:04d}.pth')
                    model.logger.save_infos(
                        epi=epi, reward=reward, step=step, alpha=model.alpha,
                        temporal_weight=model.temporal_weight,
                        critic_loss=losses[0] / max(episode_updates, 1),
                        policy_loss=losses[1] / max(episode_updates, 1))
                    if model.train_diag_interval and epi % model.train_diag_interval == 0:
                        model._append_train_diagnostics(model._episode_diagnostics(epi, reward, infos))
                    if diagnostics:
                        model._append_exploration_diagnostics(epi=epi, diagnostics=diagnostics)
                        timing = np.array([d['actor_period_ms'] for d in diagnostics])
                        path = f'{model.logger.folder_path}/async_timing.csv'
                        exists = os.path.exists(path)
                        with open(path, 'a', newline='') as stream:
                            writer = csv.writer(stream)
                            if not exists:
                                writer.writerow(['episode', 'steps', 'actor_period_mean_ms',
                                                 'actor_period_p95_ms', 'actor_period_max_ms',
                                                 'learner_updates'])
                            writer.writerow([epi, step, timing.mean(), np.percentile(timing, 95),
                                             timing.max(), updates])
                    if (epi - model.last_epi) % 100 == 0:
                        model.buffer.save_buffer(model.logger.buffer_path)
                print(f'[ ASYNC] episode={epi} steps={step} reward={reward:.4f} updates={updates}', flush=True)
                infos, diagnostics = [], []
                losses[:] = 0
                episode_updates = 0
            elif kind == 'warmup_end' and model.log:
                model.buffer.save_buffer(model.logger.buffer_path)
        if model.log:
            model.save_model(model.logger.model_path)
            model.buffer.save_buffer(model.logger.buffer_path)
        _publish(model.policy, shared, lock, version)
        status.send(('done', dict(updates=updates, buffer_size=len(model.buffer), steps=model.total_steps)))
    except BaseException:
        status.send(('error', traceback.format_exc()))
    finally:
        status.close()


class AsyncLearner:
    def __init__(self, model, queue_capacity=4096, sync_every=5):
        if queue_capacity < 1 or sync_every < 1:
            raise ValueError('queue_capacity and sync_every must be positive')
        ctx = mp.get_context('spawn')
        # Pickle bytes deliberately: multiprocessing's Tensor reducers otherwise
        # share storage with the actor, including parameters being optimized.
        learner = copy.copy(model)
        learner.env = SimpleNamespace(dim_ref_traj=model.env.dim_ref_traj,
                                      dim_policy_act=getattr(model.env, 'dim_policy_act', model.action_dim))
        payload = pickle.dumps(learner)
        self.experience = ctx.Queue(maxsize=queue_capacity)
        self.status, child_status = ctx.Pipe(duplex=False)
        values = _policy_array(model.policy)
        self.shared = ctx.RawArray('f', len(values))
        np.frombuffer(self.shared, dtype=np.float32)[:] = values
        self.lock = ctx.Lock()
        self.version = ctx.RawValue('q', 0)
        self.applied_version = 0
        self.result = None
        self.failure = None
        self.process = ctx.Process(target=_learner, args=(
            payload, self.experience, child_status, self.shared, self.lock,
            self.version, sync_every), name='sac-learner')
        self.process.start()
        child_status.close()
        if not self.status.poll(60):
            self.abort()
            raise RuntimeError('Learner startup timed out')
        kind, detail = self.status.recv()
        if kind != 'ready':
            self.abort()
            raise RuntimeError(f'Learner startup failed: {detail}')

    def check(self):
        if self.failure is not None:
            raise RuntimeError(self.failure)
        if self.status.poll():
            try:
                kind, detail = self.status.recv()
            except EOFError:
                raise RuntimeError('Learner status channel closed unexpectedly') from None
            if kind == 'error':
                self.failure = f'Learner failed:\n{detail}'
                raise RuntimeError(self.failure)
            if kind == 'done':
                self.result = detail
        if not self.process.is_alive() and self.result is None:
            raise RuntimeError(f'Learner exited: {self.process.exitcode}')

    def submit(self, kind, data=None):
        self.check()
        try:
            self.experience.put_nowait((kind, data))
        except queue.Full:
            raise RuntimeError('Learner queue full; stopping control without dropping experience') from None

    def sync(self, policy):
        if not self.lock.acquire(False):
            return False
        try:
            version = self.version.value
            if version == self.applied_version:
                return False
            values = np.frombuffer(self.shared, dtype=np.float32).copy()
        finally:
            self.lock.release()
        _install_policy(policy, values)
        self.applied_version = version
        return True

    def finish(self, timeout=60):
        # Call only after hardware control has been closed.
        deadline = time.monotonic() + timeout
        while True:
            self.check()
            try:
                self.experience.put(None, timeout=0.1)
                break
            except queue.Full:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Learner drain timed out')
        while self.result is None:
            self.check()
            if time.monotonic() >= deadline:
                raise TimeoutError('Learner shutdown timed out')
            time.sleep(0.01)
        self.process.join(timeout=max(0, deadline - time.monotonic()))
        if self.process.is_alive():
            raise TimeoutError('Learner process did not exit')
        self.experience.close()
        self.experience.join_thread()
        self.status.close()
        return self.result

    def abort(self):
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        self.experience.cancel_join_thread()
        self.experience.close()
        self.status.close()


def train_async(model, episode, warmup_steps=0, warmup_deterministic=False,
                queue_capacity=4096, sync_every=5):
    """Own the hardware lifetime; close it BEFORE waiting for learner saving."""
    learner = None
    completed = False
    try:
        learner = AsyncLearner(model, queue_capacity, sync_every)
        state, _ = model.env.reset()
        for step in range(warmup_steps):
            action = (model.evaluate_action(state) if warmup_deterministic
                      else model.select_action(state))
            next_state, reward, done, _, _ = model.env.step(action)
            learner.submit('transition', ((state.copy(), action.copy(), reward, next_state.copy(), done),
                                         model.total_steps, False, None, None))
            state = next_state
            model.total_steps += 1
            if done or (step + 1) % model.horizon == 0:
                state, _ = model.env.reset()
        if warmup_steps:
            learner.submit('warmup_end')
        previous_step_start = None
        for epi in range(episode):
            episode_index = model.last_epi + epi
            if (model.reset_after_warmup and episode_index == model.start_epi
                    and hasattr(model.env, 'request_hard_reset')):
                model.env.request_hard_reset()
            state, _ = model.env.reset()
            reward_sum = 0.0
            for _ in range(model.horizon):
                step_start = time.monotonic()
                learner.check()
                learner.sync(model.policy)
                in_warmup = model.start_epi > episode_index
                action, diag = (model._warmup_action() if in_warmup
                                else model.select_action_with_diagnostics(state))
                next_state, reward, done, _, info = model.env.step(action)
                diag['actor_period_ms'] = 1000 * (
                    step_start - previous_step_start if previous_step_start is not None
                    else time.monotonic() - step_start)
                previous_step_start = step_start
                learner.submit('transition', ((state.copy(), action.copy(), reward, next_state.copy(), done),
                                             model.total_steps, model.update_during_warmup or not in_warmup,
                                             copy.deepcopy(info) if model.log else None, diag))
                state = next_state
                model.total_steps += 1
                reward_sum += float(reward)
                if done:
                    break
            learner.submit('episode', (episode_index + 1, reward_sum, model.total_steps))
        completed = True
    finally:
        try:
            model.env.close()
        finally:
            if learner is not None:
                try:
                    result = learner.finish()
                    learner.sync(model.policy)
                finally:
                    learner.abort()
    if completed:
        return result
