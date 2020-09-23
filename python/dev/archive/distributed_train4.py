import tensorflow as tf
import unetsc
from custom_loss_classes import DiceLoss, WeightedBinaryCrossEntropy

tf.config.experimental_run_functions_eagerly(True)

class Train(object):

    def __init__(self, epochs, model, batch_size, strategy):
        self.epochs = epochs
        self.batch_size = batch_size
        self.strategy = strategy
        #self.loss_object = tf.keras.losses.BinaryCrossentropy(logits=False, reduction=tf.keras.losses.Reduction.NONE)
        self.loss_object = WeightedBinaryCrossEntropy(7)
        self.optimizer = tf.keras.optimizers.Adam()
        self.model = model


    def computeLoss(self, yPred, yTrue):

        #loss = self.loss_object(yPred, yTrue)
        loss = tf.reduce_sum(self.loss_object(yPred, yTrue)) * (1./batch_size)
        print('loss is here', loss)
        loss = loss * (1. / self.strategy.num_replicas_in_sync)

        return loss



    def train_step(self, inputs):

        image, label = inputs

        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.float32)

        with tf.GradientTape() as tape:
            predictions = self.model(image, training=True)
            loss = self.computeLoss(label, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss.numpy()


    def test_step(self, inputs):

        x, y = inputs
        predictions = self.model(image, training=False)
        loss = self.loss_object(y, logits)

        return loss


    def custom_loop(self, train_dist_dataset, test_dist_dataset, strategy):

        def distributed_train_epoch(ds):

            total_loss = 0.0

            for one_batch in ds:
                per_replica_loss = strategy.run(self.train_step, args=(one_batch,))
                total_loss += strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)


            return total_loss


        def distributed_test_epoch(ds):
            total_loss = 0.0
            for one_batch in ds:
                loss = strategy.run(self.test_step, args=(one_batch,))
                total_loss += strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

            return total_loss


        distributed_train_epoch = tf.function(distributed_train_epoch)
        distributed_test_epoch = tf.function(distributed_test_epoch)

        train_total_loss = distributed_train_epoch(train_dist_dataset)
        test_total_loss  = distributed_test_epoch(test_dist_dataset)

        template = ('Epoch: {}, Train Loss: {}, Train Accuracy: {}, Test Loss: {}, Test Accuracy: {}')

        print(template.format(epoch, train_total_loss / trainSteps, test_total_loss / validSteps))


def main(epochs, batch_size, imgDims, filters, trainDataset, testDataset, num_gpu=2):

    devices = ['/device:GPU:{}'.format(i) for i in range(num_gpu)]
    strategy = tf.distribute.MirroredStrategy(devices)

    with strategy.scope():
        print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        model = unetsc.UnetSC(filters=filters)
        model.build((1, imgDims, imgDims, 3))
        trainer = Train(epochs, model, batch_size, strategy)

        train_dist_dataset = strategy.experimental_distribute_dataset(trainDataset)
        test_dist_dataset = strategy.experimental_distribute_dataset(testDataset)

        trainer.custom_loop(train_dist_dataset, test_dist_dataset, strategy)

        return None, None
