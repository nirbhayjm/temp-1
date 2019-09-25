import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as ds


class PPO():
    def __init__(self,
                 actor_critic,
                 ppo_version,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_entropy_reg=False,
                 use_max_ent=False,
                 max_ent_action_logprob_coeff=0.001,
                 model='cond'):

        self.actor_critic = actor_critic
        self.ppo_version = ppo_version

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_entropy_reg = use_entropy_reg
        self.use_max_ent = use_max_ent
        self.model = model

        if ppo_version == 'default':
            self.optimizer = optim.Adam(
                actor_critic.parameters(), lr=lr, eps=eps)

        elif ppo_version == 'ppo-with-options':
            options_params = {name:param for name, param in \
                actor_critic.options_decoder.named_parameters()}
            # options_params_names = [name for name, param in options_params]

            actor_critic_params = {name:param for name, param \
                in actor_critic.named_parameters() \
                if name.replace('options_decoder.','') not in options_params.keys()}

            # print("options params names:",
            #     [name for name,val in options_params.items()])
            # print("actor_critic params names:",
            #     [name for name,val in actor_critic_params.items()])

            self.options_optim = optim.Adam(
                # [val for name, val in options_params],
                options_params.values(),
                lr=lr, eps=eps)
            self.actor_critic_optim = optim.Adam(
                # [val for name, val in actor_critic_params],
                actor_critic_params.values(),
                lr=lr, eps=eps)

        else:
            raise ValueError("ppo-version: {}".format(ppo_version))

    def update(self, rollouts, use_max_ent=False,
            kl_coeff=None, model='hier', q_dist=None, anneal_coeff=None,
            closed_loop=False, cl_args=None):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0


        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                return_batch, masks_batch, old_action_log_probs_batch, \
                adv_targ, omega_option_batch = sample


                # If using max-ent, do not compute entropy
                # Reshape to do in a single forward pass for all steps
                # [NOTE] : This has to be changed.
                values, action_log_probs, dist_entropy, states = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch,
                    masks_batch, actions_batch, omega_option_batch,
                    get_entropy=True)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio,
                                    1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ

                surr1 = torch.masked_select(surr1, torch.eq(masks_batch, 1))
                surr2 = torch.masked_select(surr2, torch.eq(masks_batch, 1))

                action_loss = -torch.min(surr1, surr2).mean()
                value_target = return_batch

                values = torch.masked_select(values, \
                        torch.eq(masks_batch, 1))

                value_target = torch.masked_select(value_target, \
                        torch.eq(masks_batch, 1))

                value_loss = F.mse_loss(values, value_target)

                if self.ppo_version == 'ppo-with-options':
                    self.options_optim.zero_grad()
                    self.actor_critic_optim.zero_grad()
                else:
                    self.optimizer.zero_grad()

                if not self.use_entropy_reg:
                    final_loss = (value_loss * self.value_loss_coef + \
                                 action_loss)
                else:
                    final_loss = (value_loss * self.value_loss_coef + \
                                 action_loss - \
                                 dist_entropy * self.entropy_coef)


                final_loss.backward(retain_graph=True)

                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)

                if self.ppo_version == 'ppo-with-options':
                    self.actor_critic_optim.step()
                else:
                    self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        if self.ppo_version == 'ppo-with-options':
            opt_action_loss, opt_dist_entropy, KLD, kl_loss,\
            p_entropy, q_entropy= \
                self.options_decoder_update(rollouts=rollouts,
                                            kl_coeff=kl_coeff,
                                            q_dist=q_dist,
                                            model=model,
                                            anneal_coeff=anneal_coeff,
                                            closed_loop=closed_loop,
                                            cl_args=cl_args)

            option_rets = {
                'opt_action_loss': opt_action_loss,
                'opt_dist_entropy': opt_dist_entropy,
                'KLD': KLD,
                'kl_loss': kl_loss,
                'p_entropy': p_entropy,
                'q_entropy': q_entropy,
            }

            return value_loss_epoch, action_loss_epoch, dist_entropy_epoch,\
                option_rets

        else:
            return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def options_decoder_update(self, rollouts, kl_coeff, q_dist,
            model, anneal_coeff, closed_loop=False, cl_args=None):
        assert self.ppo_version == 'ppo-with-options'

        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        _, omega_option_dim = rollouts.omega_option.size()
        omega_option = rollouts.omega_option.unsqueeze(0).repeat(num_steps, 1, 1)
        omega_option = omega_option.view(-1, omega_option_dim)

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            inputs=rollouts.obs[:-1].view(-1, *obs_shape),
            rnn_hxs=rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            masks=rollouts.masks[:-1].view(-1, 1),
            action=rollouts.actions.view(-1, action_shape),
            omega_option=omega_option,
            get_entropy=True)

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        # value_target = rollouts.returns[:-1]

        # value_loss = advantages.pow(2).mean()
        # value_loss = F.mse_loss(values, value_target.detach())

        action_loss = -(advantages.detach() * action_log_probs).mean()

        if model == 'hier':
            if closed_loop:
                trajectory = cl_args['trajectory']
                trajectory_encoder = cl_args['trajectory_encoder']
                trajectory_optim = cl_args['trajectory_optim']
                p_dist = trajectory_encoder(trajectory)
            else:
                p_dist = ds.normal.Normal(
                    loc=q_dist.loc.new_zeros(q_dist.loc.shape),
                    scale=q_dist.scale.new_ones(q_dist.scale.shape)
                )

            KLD = ds.kl.kl_divergence(q_dist, p_dist)
            KLD = torch.sum(KLD, 1).mean()

            # anneal_coeff = utils.kl_coefficient_curriculum(
            #     iter_id=iter_id,
            #     iters_per_epoch=num_batches_per_epoch,
            #     start_after_epochs=args.kl_anneal_start_epochs,
            #     linear_growth_epochs=args.kl_anneal_growth_epochs,
            # )

            kl_loss = KLD * kl_coeff * anneal_coeff

            # if kl_coeff * anneal_coeff > 0.0:
            #     agent.optimizer.zero_grad()
            #     kl_loss.backward(retain_graph=False)
            #     agent.optimizer.step()

            with torch.no_grad():
                p_entropy = p_dist.entropy().mean().item()
                q_entropy = q_dist.entropy().mean().item()

            KLD_ret = KLD.item()
            kl_loss_ret = kl_loss.item()

        else:
            # Conditional model
            kl_loss = 0.0
            KLD_ret = None
            kl_loss_ret = None
            p_entropy = None
            q_entropy = None

        final_loss = kl_loss + action_loss
        if self.use_entropy_reg:
            final_loss = final_loss - dist_entropy * self.entropy_coef

        self.options_optim.zero_grad()
        if model == 'hier' and closed_loop:
            trajectory_optim.zero_grad()

        final_loss.backward()

        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                 self.max_grad_norm)

        self.options_optim.step()
        if model == 'hier' and closed_loop:
            trajectory_optim.step()

        return action_loss.item(), dist_entropy.item(),\
            KLD_ret, kl_loss_ret, p_entropy, q_entropy
