import tensorflow as tf


def positional_embedding(pos_seq, inv_freq, bsz=None):
    sinusoid_inp = tf.einsum('i,j->ij', pos_seq, inv_freq)
    pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
    if bsz is not None:
        return tf.tile(pos_emb[:, None, :], [1, bsz, 1])
    else:
        return pos_emb[:, None, :]


def positionwise_FF(inp, d_model, d_inner, dropout, kernel_initializer,
                    scope='ff', is_training=True, layer_norm=True):
    output = inp
    with tf.variable_scope(scope):
        output = tf.layers.dense(inp, d_inner, activation=tf.nn.relu,
                                 kernel_initializer=kernel_initializer,
                                 name='layer_1')
        output = tf.layers.dropout(output, dropout, training=is_training,
                                   name='drop_1')
        output = tf.layers.dense(output, d_model,
                                 kernel_initializer=kernel_initializer,
                                 name='layer_2')
        output = tf.layers.dropout(output, dropout, training=is_training,
                                   name='drop_2')
        if layer_norm:
            output = tf.contrib.layers.layer_norm(output + inp, begin_norm_axis=-1)
    return output


def rel_shift(x):
    x_size = tf.shape(x)

    x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
    x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, x_size)

    return x


def rel_multihead_attn(w, r, r_w_bias, r_r_bias, attn_mask, mems, d_model,
                       n_head, d_head, dropout, dropatt, is_training,
                       kernel_initializer, scope='rel_attn'):
    scale = 1 / (d_head ** 0.5)
    with tf.variable_scope(scope):
        qlen = tf.shape(w)[0]
        rlen = tf.shape(r)[0]
        bsz = tf.shape(w)[1]

        cat = tf.concat([mems, w],
                        0) if mems is not None and mems.shape.ndims > 1 else w
        w_heads = tf.layers.dense(cat, 3 * n_head * d_head, use_bias=False,
                                  kernel_initializer=kernel_initializer, name='qkv')
        r_head_k = tf.layers.dense(r, n_head * d_head, use_bias=False,
                                   kernel_initializer=kernel_initializer, name='r')

        w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
        w_head_q = w_head_q[-qlen:]

        klen = tf.shape(w_head_k)[0]

        w_head_q = tf.reshape(w_head_q, [qlen, bsz, n_head, d_head])
        w_head_k = tf.reshape(w_head_k, [klen, bsz, n_head, d_head])
        w_head_v = tf.reshape(w_head_v, [klen, bsz, n_head, d_head])

        r_head_k = tf.reshape(r_head_k, [rlen, n_head, d_head])

        rw_head_q = w_head_q + r_w_bias
        rr_head_q = w_head_q + r_r_bias

        AC = tf.einsum('ibnd,jbnd->ijbn', rw_head_q, w_head_k)
        BD = tf.einsum('ibnd,jnd->ijbn', rr_head_q, r_head_k)
        BD = rel_shift(BD)

        attn_score = (AC + BD) * scale
        attn_mask_t = attn_mask[:, :, None, None]
        attn_score = attn_score * (1 - attn_mask_t) - 1e30 * attn_mask_t

        attn_prob = tf.nn.softmax(attn_score, 1)
        attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)

        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
        size_t = tf.shape(attn_vec)
        attn_vec = tf.reshape(attn_vec, [size_t[0], size_t[1], n_head * d_head])

        attn_out = tf.layers.dense(attn_vec, d_model, use_bias=False,
                                   kernel_initializer=kernel_initializer, name='o')
        attn_out = tf.layers.dropout(attn_out, dropout, training=is_training)

        output = tf.contrib.layers.layer_norm(attn_out + w, begin_norm_axis=-1)
    return output


def embedding_lookup(lookup_table, x, use_tpu=True):
    if use_tpu:
        n_token = tf.shape(lookup_table)[0]
        one_hot_idx = tf.one_hot(x, n_token)
        if one_hot_idx.shape.ndims == 2:
            return tf.einsum('nd,in->id', lookup_table, one_hot_idx)
        else:
            return tf.einsum('nd,ibn->ibd', lookup_table, one_hot_idx)
    else:
        return tf.nn.embedding_lookup(lookup_table, x)


def mask_adaptive_embedding_lookup(x, n_token, d_embed, d_proj, cutoffs, initializer,
                                   proj_initializer, div_val=1,
                                   proj_same_dim=True,
                                   scope='adaptive_embed', **kwargs):
    emb_scale = d_proj ** 0.5
    with tf.variable_scope(scope):
        if div_val == 1:
            lookup_table = tf.get_variable('lookup_table', [n_token, d_embed],
                                           initializer=initializer)
            y = embedding_lookup(lookup_table, x, use_tpu=False)
            if d_proj != d_embed:
                proj_W = tf.get_variable('proj_W', [d_embed, d_proj],
                                         initializer=proj_initializer)
                y = tf.einsum('ibe,ed->ibd', y, proj_W)
            else:
                proj_W = None
            ret_params = [lookup_table, proj_W]
        else:
            tables, projs = [], []
            cutoff_ends = [0] + cutoffs + [n_token]
            x_size = tf.shape(x)
            y = tf.zeros([x_size[0], x_size[1], d_proj])
            for i in range(len(cutoff_ends) - 1):
                with tf.variable_scope('cutoff_{}'.format(i)):
                    l_idx, r_idx = cutoff_ends[i], cutoff_ends[i + 1]
                    mask = (x >= l_idx) & (x < r_idx)
                    cur_x = tf.boolean_mask(x, mask) - l_idx
                    cur_d_embed = d_embed // (div_val ** i)
                    lookup_table = tf.get_variable('lookup_table',
                                                   [r_idx - l_idx, cur_d_embed],
                                                   initializer=initializer)
                    cur_y = embedding_lookup(lookup_table, cur_x, use_tpu=False)
                    if d_proj == cur_d_embed and not proj_same_dim:
                        proj_W = None
                    else:
                        proj_W = tf.get_variable('proj_W', [cur_d_embed, d_proj],
                                                 initializer=proj_initializer)
                        cur_y = tf.einsum('id,de->ie', cur_y, proj_W)
                    mask_idx = tf.to_int64(tf.where(mask))
                    y += tf.scatter_nd(mask_idx, cur_y, tf.to_int64(tf.shape(y)))
                    tables.append(lookup_table)
                    projs.append(proj_W)
            ret_params = [tables, projs]

    y *= emb_scale
    return y, ret_params


def mul_adaptive_embedding_lookup(x, n_token, d_embed, d_proj, cutoffs, initializer,
                                  proj_initializer, div_val=1, perms=None,
                                  proj_same_dim=True,
                                  scope='adaptive_embed'):
    """
    perms: If None, first compute W = W1 x W2 (projection for each bin),
        and then compute X x W (embedding lookup). If not None,
        use bin-based embedding lookup with max_bin_size defined by
        the shape of perms.
    """
    emb_scale = d_proj ** 0.5
    with tf.variable_scope(scope):
        if div_val == 1:
            lookup_table = tf.get_variable('lookup_table', [n_token, d_embed],
                                           initializer=initializer)
            y = embedding_lookup(lookup_table, x)
            if d_proj != d_embed:
                proj_W = tf.get_variable('proj_W', [d_embed, d_proj],
                                         initializer=proj_initializer)
                y = tf.einsum('ibe,ed->ibd', y, proj_W)
            else:
                proj_W = None
            ret_params = [lookup_table, proj_W]
        else:
            tables, projs = [], []
            cutoff_ends = [0] + cutoffs + [n_token]
            x_size = tf.shape(x)
            if perms is None:
                cat_lookup = []
            else:
                cat_lookup = tf.zeros([x_size[0], x_size[1], d_proj])
            for i in range(len(cutoff_ends) - 1):
                with tf.variable_scope('cutoff_{}'.format(i)):
                    l_idx, r_idx = cutoff_ends[i], cutoff_ends[i + 1]
                    cur_d_embed = d_embed // (div_val ** i)
                    lookup_table = tf.get_variable('lookup_table',
                                                   [r_idx - l_idx, cur_d_embed],
                                                   initializer=initializer)
                    if cur_d_embed == d_proj and not proj_same_dim:
                        proj_W = None
                    else:
                        proj_W = tf.get_variable('proj_W', [cur_d_embed, d_proj],
                                                 initializer=proj_initializer)
                    if perms is None:
                        cat_lookup.append(tf.einsum('ie,ed->id', lookup_table, proj_W))
                    else:
                        # speed up the computation of the first bin
                        # also save some meory
                        if i == 0:
                            cur_y = embedding_lookup(lookup_table, tf.minimum(x, r_idx - 1))
                            if proj_W is not None:
                                cur_y = tf.einsum('ibe,ed->ibd', cur_y, proj_W)
                            cur_y *= perms[i][:, :, None]
                            cat_lookup += cur_y
                        else:
                            cur_x = tf.einsum('ib,ibk->k', tf.to_float(x - l_idx), perms[i])
                            cur_x = tf.to_int32(cur_x)
                            cur_y = embedding_lookup(lookup_table, cur_x)
                            if proj_W is not None:
                                cur_y = tf.einsum('ke,ed->kd', cur_y, proj_W)
                            cat_lookup += tf.einsum('kd,ibk->ibd', cur_y, perms[i])
                    tables.append(lookup_table)
                    projs.append(proj_W)
            if perms is None:
                cat_lookup = tf.concat(cat_lookup, 0)
                y = embedding_lookup(cat_lookup, x)
            else:
                y = cat_lookup
            ret_params = [tables, projs]

    y *= emb_scale
    return y, ret_params


def mask_adaptive_logsoftmax(hidden, target, n_token, d_embed, d_proj, cutoffs,
                             params, tie_projs,
                             initializer=None, proj_initializer=None,
                             div_val=1, scope='adaptive_softmax',
                             proj_same_dim=True,
                             return_mean=True, unk=False, **kwargs):
    def _logit(x, W, b, proj):
        y = x
        if proj is not None:
            y = tf.einsum('ibd,ed->ibe', y, proj)
        return tf.einsum('ibd,nd->ibn', y, W) + b

    params_W, params_projs = params[0], params[1]

    def _gather_logprob(logprob, target):
        lp_size = tf.shape(logprob)
        r = tf.range(lp_size[0])
        idx = tf.stack([r, target], 1)
        return tf.gather_nd(logprob, idx)

    with tf.variable_scope(scope):
        if len(cutoffs) == 0:
            softmax_b = tf.get_variable('bias', [n_token],
                                        initializer=tf.zeros_initializer())
            output = _logit(hidden, params_W, softmax_b, params_projs)
            logit = tf.reshape(output, [-1, n_token])
            probs = tf.nn.softmax(logit)
            labels = tf.reshape(target, [-1])
            # weights = tf.ones(labels.shape, dtype=tf.float32)

            if unk:
                unk_id = n_token - 2
                unk_tf = tf.constant(value=unk_id, dtype=tf.int32, shape=labels.shape)
                # zero_weights = tf.zeros_like(labels, dtype=tf.float32)
                wrong_label = tf.constant(value=-1, dtype=tf.int32, shape=labels.shape)

                condition_tf = tf.equal(labels, unk_tf)
                # new_weights = tf.where(condition_tf, zero_weights, weights)
                new_labels = tf.where(condition_tf, wrong_label, labels)
            else:
                # new_weights = weights
                new_labels = labels

            correct_prediction = tf.equal(tf.cast(tf.argmax(probs, 1), dtype=tf.int32), new_labels)

            nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
                                                                 logits=output)



        else:
            cutoff_ends = [0] + cutoffs + [n_token]
            nll = tf.zeros_like(target, dtype=tf.float32)
            for i in range(len(cutoff_ends) - 1):
                with tf.variable_scope('cutoff_{}'.format(i)):
                    l_idx, r_idx = cutoff_ends[i], cutoff_ends[i + 1]
                    mask = (target >= l_idx) & (target < r_idx)
                    mask_idx = tf.where(mask)
                    cur_target = tf.boolean_mask(target, mask) - l_idx
                    cur_d_embed = d_embed // (div_val ** i)

                    if div_val == 1:
                        cur_W = params_W[l_idx: r_idx]
                    else:
                        cur_W = params_W[i]
                    cur_b = tf.get_variable('b', [r_idx - l_idx],
                                            initializer=tf.zeros_initializer())
                    if tie_projs[i]:
                        if div_val == 1:
                            cur_proj = params_projs
                        else:
                            cur_proj = params_projs[i]
                    else:
                        if (div_val == 1 or not proj_same_dim) and d_proj == cur_d_embed:
                            cur_proj = None
                        else:
                            cur_proj = tf.get_variable('proj', [cur_d_embed, d_proj],
                                                       initializer=proj_initializer)
                    if i == 0:
                        cluster_W = tf.get_variable('cluster_W', [len(cutoffs), d_embed],
                                                    initializer=tf.zeros_initializer())
                        cluster_b = tf.get_variable('cluster_b', [len(cutoffs)],
                                                    initializer=tf.zeros_initializer())
                        cur_W = tf.concat([cur_W, cluster_W], 0)
                        cur_b = tf.concat([cur_b, cluster_b], 0)

                        head_logit = _logit(hidden, cur_W, cur_b, cur_proj)
                        head_logprob = tf.nn.log_softmax(head_logit)
                        cur_head_logprob = tf.boolean_mask(head_logprob, mask)
                        cur_logprob = _gather_logprob(cur_head_logprob, cur_target)
                    else:
                        cur_head_logprob = tf.boolean_mask(head_logprob, mask)
                        cur_hidden = tf.boolean_mask(hidden, mask)
                        tail_logit = tf.squeeze(_logit(
                            cur_hidden[None], cur_W, cur_b, cur_proj), 0)
                        tail_logprob = tf.nn.log_softmax(tail_logit)
                        cur_logprob = (cur_head_logprob[:, cutoff_ends[1] + i - 1] +
                                       _gather_logprob(tail_logprob, cur_target))
                    nll += tf.scatter_nd(mask_idx, -cur_logprob,
                                         tf.to_int64(tf.shape(nll)))
    if return_mean:
        nll = tf.reduce_mean(nll)
    return nll, correct_prediction


def mul_adaptive_logsoftmax(hidden, target, n_token, d_embed, d_proj, cutoffs,
                            params, tie_projs,
                            initializer=None, proj_initializer=None,
                            div_val=1, perms=None, proj_same_dim=True,
                            scope='adaptive_softmax',
                            **kwargs):
    def _logit(x, W, b, proj):
        y = x
        if x.shape.ndims == 3:
            if proj is not None:
                y = tf.einsum('ibd,ed->ibe', y, proj)
            return tf.einsum('ibd,nd->ibn', y, W) + b
        else:
            if proj is not None:
                y = tf.einsum('id,ed->ie', y, proj)
            return tf.einsum('id,nd->in', y, W) + b

    params_W, params_projs = params[0], params[1]

    with tf.variable_scope(scope):
        if len(cutoffs) == 0:
            softmax_b = tf.get_variable('bias', [n_token],
                                        initializer=tf.zeros_initializer())
            output = _logit(hidden, params_W, softmax_b, params_projs)
            nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
                                                                 logits=output)
            nll = tf.reduce_mean(nll)
        else:
            total_loss, total_cnt = 0, 0
            cutoff_ends = [0] + cutoffs + [n_token]
            for i in range(len(cutoff_ends) - 1):
                with tf.variable_scope('cutoff_{}'.format(i)):
                    l_idx, r_idx = cutoff_ends[i], cutoff_ends[i + 1]

                    cur_d_embed = d_embed // (div_val ** i)

                    if div_val == 1:
                        cur_W = params_W[l_idx: r_idx]
                    else:
                        cur_W = params_W[i]
                    cur_b = tf.get_variable('b', [r_idx - l_idx],
                                            initializer=tf.zeros_initializer())
                    if tie_projs[i]:
                        if div_val == 1:
                            cur_proj = params_projs
                        else:
                            cur_proj = params_projs[i]
                    else:
                        if (div_val == 1 or not proj_same_dim) and d_proj == cur_d_embed:
                            cur_proj = None
                        else:
                            cur_proj = tf.get_variable('proj', [cur_d_embed, d_proj],
                                                       initializer=proj_initializer)

                    if i == 0:
                        cluster_W = tf.get_variable('cluster_W', [len(cutoffs), d_embed],
                                                    initializer=tf.zeros_initializer())
                        cluster_b = tf.get_variable('cluster_b', [len(cutoffs)],
                                                    initializer=tf.zeros_initializer())
                        cur_W = tf.concat([cur_W, cluster_W], 0)
                        cur_b = tf.concat([cur_b, cluster_b], 0)

                        head_logit = _logit(hidden, cur_W, cur_b, cur_proj)

                        head_target = kwargs.get("head_target")
                        head_nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=head_target,
                            logits=head_logit)

                        masked_loss = head_nll * perms[i]
                        total_loss += tf.reduce_sum(masked_loss)
                        total_cnt += tf.reduce_sum(perms[i])

                        # head_logprob = tf.nn.log_softmax(head_logit)

                        # final_logprob = head_logprob * perms[i][:, :, None]
                        # final_target = tf.one_hot(target, tf.shape(head_logprob)[2])
                        # total_loss -= tf.einsum('ibn,ibn->', final_logprob, final_target)
                        # total_cnt += tf.reduce_sum(perms[i])
                    else:
                        cur_head_nll = tf.einsum('ib,ibk->k', head_nll, perms[i])

                        cur_hidden = tf.einsum('ibd,ibk->kd', hidden, perms[i])
                        tail_logit = _logit(cur_hidden, cur_W, cur_b, cur_proj)

                        tail_target = tf.einsum('ib,ibk->k', tf.to_float(target - l_idx),
                                                perms[i])
                        tail_nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=tf.to_int32(tail_target),
                            logits=tail_logit)

                        sum_nll = cur_head_nll + tail_nll
                        mask = tf.reduce_sum(perms[i], [0, 1])

                        masked_loss = sum_nll * mask
                        total_loss += tf.reduce_sum(masked_loss)
                        total_cnt += tf.reduce_sum(mask)

            nll = total_loss / total_cnt

    return nll


def _create_mask(qlen, mlen, same_length=False):
    attn_mask = tf.ones([qlen, qlen])
    mask_u = tf.matrix_band_part(attn_mask, 0, -1)
    mask_dia = tf.matrix_band_part(attn_mask, 0, 0)
    attn_mask_pad = tf.zeros([qlen, mlen])
    ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
    if same_length:
        mask_l = tf.matrix_band_part(attn_mask, -1, 0)
        ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
    return ret


def _cache_mem(curr_out, prev_mem, mem_len=None):
    if mem_len is None or prev_mem is None:
        new_mem = curr_out
    elif mem_len == 0:
        return prev_mem
    else:
        new_mem = tf.concat([prev_mem, curr_out], 0)[- mem_len:]

    return tf.stop_gradient(new_mem)


def encode_par_path(embedding_inputs, parent_hidden_size, rnn_layers=1, keep_prob=0.7, bi_lstm=True):
    with tf.variable_scope('path_encoder') as encoder_scope:
        def build_cell(hidden_size):
            def get_single_cell(hidden_size, keep_prob):
                cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
                if keep_prob < 1:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
                return cell

            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_single_cell(hidden_size, keep_prob) for _ in range(rnn_layers)])

            return cell

        if not bi_lstm:
            encoder_cell = build_cell(parent_hidden_size)
            encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                encoder_cell, embedding_inputs,
                # sequence_length=self.par_seq_len,
                dtype=tf.float32, scope=encoder_scope)
            return encoder_outputs, encoder_final_state
        else:
            encoder_cell = build_cell(parent_hidden_size / 2)
            bw_encoder_cell = build_cell(parent_hidden_size / 2)
            encoder_outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
                encoder_cell, bw_encoder_cell,
                embedding_inputs,
                # sequence_length=self.par_seq_len,
                dtype=tf.float32, scope=encoder_scope)

            state = []
            for i in range(rnn_layers):
                fs = fw_state[i]
                bs = bw_state[i]
                encoder_final_state_c = tf.concat((fs.c, bs.c), 1)
                encoder_final_state_h = tf.concat((fs.h, bs.h), 1)
                encoder_final_state = tf.nn.rnn_cell.LSTMStateTuple(
                    c=encoder_final_state_c,
                    h=encoder_final_state_h)
                state.append(encoder_final_state)
            encoder_final_state = tuple(state)

            encoder_outputs = tf.concat([encoder_outputs[0], encoder_outputs[1]], -1)
            return encoder_outputs, encoder_final_state


def transformer(inpN, inpT, targetsN, inputsPath, parent_hidden_size, num_steps, mems, n_token_N, n_token_T,
                n_layer, d_model_N, d_model_T, d_embed_N, d_embed_T,
                n_head, d_head, d_inner, dropout, dropatt,
                initializer, is_training, proj_initializer=None,
                mem_len=None, cutoffs=[], div_val=1, tie_projs=[],
                same_length=False, clamp_len=-1, use_tpu=True,
                input_perms=None, target_perms=None, head_target=None,
                untie_r=False, proj_same_dim=True,
                scope='transformer'):
    """
    cutoffs: a list of python int. Cutoffs for adaptive softmax.
    tie_projs: a list of python bools. Whether to tie the projections.
    use_tpu: if True, use one_hot in embedding lookup and bin-based implementation
          of adaptive softmax.
    perms: a list of tensors. Each tensor should of size [len, bsz, bin_size].
          Only used in the adaptive setting.
    """
    new_mems = []
    with tf.variable_scope(scope):
        if untie_r:
            r_w_bias = tf.get_variable('r_w_bias', [n_layer, n_head, d_head],
                                       initializer=initializer)
            r_r_bias = tf.get_variable('r_r_bias', [n_layer, n_head, d_head],
                                       initializer=initializer)
        else:
            r_w_bias = tf.get_variable('r_w_bias', [n_head, d_head],
                                       initializer=initializer)
            r_r_bias = tf.get_variable('r_r_bias', [n_head, d_head],
                                       initializer=initializer)

        qlen = tf.shape(inpN)[0]
        mlen = tf.shape(mems[0])[0] if mems is not None else 0
        klen = mlen + qlen

        if proj_initializer is None:
            proj_initializer = initializer

        lookup_fn_N = (mul_adaptive_embedding_lookup if use_tpu else
                       mask_adaptive_embedding_lookup)
        lookup_fn_T = (mul_adaptive_embedding_lookup if use_tpu else
                       mask_adaptive_embedding_lookup)

        embeddingsN, shared_paramsN = lookup_fn_N(
            x=inpN,
            n_token=n_token_N,
            d_embed=d_embed_N,
            d_proj=d_model_N,
            scope='adaptive_embed_N',
            cutoffs=cutoffs,
            initializer=initializer,
            proj_initializer=proj_initializer,
            div_val=div_val,
            perms=input_perms,
            proj_same_dim=proj_same_dim)

        embeddingsT, shared_paramsT = lookup_fn_T(
            x=inpT,
            n_token=n_token_T,
            d_embed=d_embed_T,
            d_proj=d_model_T,
            scope='adaptive_embed_T',
            cutoffs=cutoffs,
            initializer=initializer,
            proj_initializer=proj_initializer,
            div_val=div_val,
            perms=input_perms,
            proj_same_dim=proj_same_dim)

        inputsPath, _ = lookup_fn_N(
            x=inputsPath,
            n_token=n_token_N,
            d_embed=d_embed_N,
            d_proj=d_model_N,
            scope='adaptive_embed_Path',
            cutoffs=cutoffs,
            initializer=initializer,
            proj_initializer=proj_initializer,
            div_val=div_val,
            perms=input_perms,
            proj_same_dim=proj_same_dim)

        embeddings = tf.concat([embeddingsN, embeddingsT], 2)

        attn_mask = _create_mask(qlen, mlen, same_length)

        pos_seq = tf.range(klen - 1, -1, -1.0)
        if clamp_len > 0:
            pos_seq = tf.minimum(pos_seq, clamp_len)

        d_model = d_model_T + d_model_N
        inv_freq = 1 / (10000 ** (tf.range(0, d_model, 2.0) / d_model))
        pos_emb = positional_embedding(pos_seq, inv_freq)

        output = tf.layers.dropout(embeddings, dropout, training=is_training)
        pos_emb = tf.layers.dropout(pos_emb, dropout, training=is_training)

        if mems is None:
            mems = [None] * n_layer

        # Path2root
        root_path = []
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                path_output, path_state = encode_par_path(
                    inputsPath[:, time_step, :, :], parent_hidden_size)  # [bz, parent_len, hidden]
                root_path.append(path_output[:, -1, :])  # [seq_len, bz, hidden]

        root_path_output = tf.stack(axis=0, values=root_path)  # [bz, seq_len, hidden]

        for i in range(n_layer):
            # cache new mems
            new_mems.append(_cache_mem(output, mems[i], mem_len))

            with tf.variable_scope('layer_{}'.format(i)):
                output = rel_multihead_attn(
                    w=output,
                    r=pos_emb,
                    r_w_bias=r_w_bias if not untie_r else r_w_bias[i],
                    r_r_bias=r_r_bias if not untie_r else r_r_bias[i],
                    attn_mask=attn_mask,
                    mems=mems[i],
                    d_model=d_model,
                    n_head=n_head,
                    d_head=d_head,
                    dropout=dropout,
                    dropatt=dropatt,
                    is_training=is_training,
                    kernel_initializer=initializer)

                output = positionwise_FF(
                    inp=output,
                    d_model=d_model,
                    d_inner=d_inner,
                    dropout=dropout,
                    kernel_initializer=initializer,
                    scope='ff',
                    is_training=is_training)

        output_path = tf.concat([output, root_path_output], axis=-1)

        outputN = positionwise_FF(
            inp=output_path,
            d_model=d_model_N,
            d_inner=d_inner,
            dropout=dropout,
            kernel_initializer=initializer,
            scope='ff_N',
            is_training=is_training,
            layer_norm=False)

        # outputT = positionwise_FF(
        #     inp=output_path,
        #     d_model=d_model_T,
        #     d_inner=d_inner,
        #     dropout=dropout,
        #     kernel_initializer=initializer,
        #     scope='ff_T',
        #     is_training=is_training,
        #     layer_norm=False)

        outputN = tf.layers.dropout(outputN, dropout, training=is_training)
        # outputT = tf.layers.dropout(outputT, dropout, training=is_training)

        logsoftmax_fn = (mul_adaptive_logsoftmax if use_tpu else
                         mask_adaptive_logsoftmax)
        lossN, correct_predictionN = logsoftmax_fn(
            hidden=outputN,
            target=targetsN,
            n_token=n_token_N,
            d_embed=d_embed_N,
            d_proj=d_model_N,
            cutoffs=cutoffs,
            params=shared_paramsN,
            tie_projs=tie_projs,
            initializer=initializer,
            proj_initializer=proj_initializer,
            div_val=div_val,
            perms=target_perms,
            head_target=head_target,
            proj_same_dim=proj_same_dim,
            scope='adaptive_softmax_N', )

        # lossT, correct_predictionT = logsoftmax_fn(
        #     hidden=outputT,
        #     target=targetsT,
        #     n_token=n_token_T,
        #     d_embed=d_embed_T,
        #     d_proj=d_model_T,
        #     cutoffs=cutoffs,
        #     params=shared_paramsT,
        #     tie_projs=tie_projs,
        #     initializer=initializer,
        #     proj_initializer=proj_initializer,
        #     div_val=div_val,
        #     perms=target_perms,
        #     head_target=head_target,
        #     proj_same_dim=proj_same_dim,
        #     scope='adaptive_softmax_T',
        #     unk=True)
        return lossN, new_mems, correct_predictionN
