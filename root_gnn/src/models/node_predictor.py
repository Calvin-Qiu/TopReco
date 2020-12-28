import tensorflow as tf
import sonnet as snt

from graph_nets import utils_tf
from graph_nets import modules
from graph_nets import blocks

from graph_nets.modules import InteractionNetwork

from root_gnn.src.datasets.topreco import n_target_node_features, n_max_tops
from root_gnn.src.models.base import MLPGraphNetwork
from root_gnn.src.models.base import make_mlp_model

LATENT_SIZE = 128

class FourTopPredictor(snt.Module):
    """
    The mdoel predicts 7 parameters for each node with only the
    leading 4 nodes are used since they are the ghost nodes serving
    as the top quark candidates.
    """
    def __init__(self, name="FourTopPredictor"):
        super(FourTopPredictor, self).__init__(name=name)

        # input encoder
        self._edge_block = blocks.EdgeBlock(
            edge_model_fn=make_mlp_model,
            use_edges=False,
            
            use_receiver_nodes=True,
            use_sender_nodes=True,
            use_globals=False,
            name='edge_encoder_block'
        )
        self._node_encoder_block = blocks.NodeBlock(
            node_model_fn=make_mlp_model,
            use_received_edges=False,
            use_sent_edges=False,
            use_nodes=True,
            use_globals=False,
            name='node_encoder_block'
        )

        self._global_block = blocks.GlobalBlock(
            global_model_fn=make_mlp_model,
            use_edges=True,
            use_nodes=True,
            use_globals=False,
        )

        # initial state encoder
        self._edge_block_init = blocks.EdgeBlock(
            edge_model_fn=make_mlp_model,
            use_edges=False,
            
            use_receiver_nodes=True,
            use_sender_nodes=True,
            use_globals=False,
            name='edge_encoder_block'
        )
        self._node_encoder_block_init = blocks.NodeBlock(
            node_model_fn=make_mlp_model,
            use_received_edges=False,
            use_sent_edges=False,
            use_nodes=True,
            use_globals=False,
            name='node_encoder_block'
        )

        self._global_block_init = blocks.GlobalBlock(
            global_model_fn=make_mlp_model,
            use_edges=True,
            use_nodes=True,
            use_globals=False,
        )

        self._core = MLPGraphNetwork()

        self._input_state = MLPGraphNetwork()


        # Transforms the outputs into appropriate shapes.
        # 4-vector and continue probability(logit)
        global_output_size = n_target_node_features
        self._global_nn = snt.nets.MLP([128, 128, global_output_size],
                        activation=tf.nn.leaky_relu, # default is relu, tanh
                        dropout_rate=0.30,
                        name='global_output'
                    )

    def encode_initial_state(self, input):
        return self._global_block_init(self._edge_block_init(self._node_encoder_block_init(input)))

    def encode_input(self, input):
        return self._global_block(self._edge_block(self._node_encoder_block(input)))


    def __call__(self, inputs_tr, num_processing_steps, is_training=True):
        encoded_input = self.encode_input(inputs_tr)
        encoded_initial_state_tr = self.encode_initial_state(inputs_tr)
        last_state = encoded_initial_state_tr
        tops_pred = []
        for _ in range(n_max_tops):
            top, last_state = self.predict_next(encoded_input, last_state, num_processing_steps, is_training)
            tops_pred.append(top)
        return tops_pred


    def predict_next(self, encoded_input, last_state, num_processing_steps, is_training=True):
        state = self._input_state(utils_tf.concat([encoded_input, last_state], axis=1))

        output_ops = []
        for _ in range(num_processing_steps):
            state = self._core(state)
            output = state.replace(globals=self._global_nn(state.globals, is_training))
            output_ops.append(output)

        return output_ops, state

def loss_fcn(target_op, output_ops):
    loss_ops = [ tf.nn.l2_loss((target_op.globals[:, :n_max_tops*4] - output_op.globals[:, :n_max_tops*4]))
        for output_op in output_ops
    ]
    return tf.stack(loss_ops)

class FourTopOptimizer(snt.Module):
    def __init__(self, model,
                batch_size=100,
                num_epochs=100,
                lr=0.001,
                decay_lr_start_epoch=20,
                decay_lr=True,
                name=None):
        super().__init__(name=name)
        self.model = model
        self.init_lr = lr
        self.lr = tf.Variable(lr, trainable=False, name='lr', dtype=tf.float32)
        self.opt = snt.optimizers.Adam(learning_rate=self.lr)
        self.num_epochs = tf.constant(num_epochs, dtype=tf.int32)
        self.decay_lr_start_epoch = tf.constant(decay_lr_start_epoch, dtype=tf.int32)
        self.decay_lr = decay_lr
    
    def step(self, inputs_tr, targets_tr, first_batch):
        pass