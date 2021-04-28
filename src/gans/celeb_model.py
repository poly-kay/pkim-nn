import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
# import PIL
from PIL import Image

from tensorflow.keras import layers
import time
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from numba import jit

from IPython import display

class CelebModel:

    def __init__(self, epochs, noise_dim, image_shape, learn_rate, load_checkpoint, checkpoint_dir, mini_batch=64, train=False, dir_data="", num_data=200000):
        self.epochs = epochs
        self.noise_dim = noise_dim
        self.mini_batch = mini_batch
        self.img_shape = image_shape
        self.learn_rate = learn_rate
        self.num_examples_to_generate = 8

        # Required for proper GPU usage
        # physical_devices = tf.config.list_physical_devices('GPU')
        # tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(tf.__version__)

        self.generator = self.__make_generator_model()
        self.discriminator = self.__make_discriminator_model()
        # This method returns a helper function to compute cross entropy loss
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(
            from_logits=True)

        # Set Adam optimizer (Momentum + RMSProp) for both generator & discriminator
        self.generator_optimizer = tf.keras.optimizers.Adam(self.learn_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            self.learn_rate)

        #self.checkpoint_dir = './gans_model/training_checkpoints/celeb'
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)
        if os.path.exists(self.checkpoint_dir):
            if (load_checkpoint):
                self.checkpoint.restore(
                    tf.train.latest_checkpoint(self.checkpoint_dir))
        
        if train:
            self.init_train(num_data=num_data, dir_data=dir_data)

    #@staticmethod
    #@jit(nopython=True)
    def format_images(self, imgs):

        # Squash images that are in [0, 255] to (-1, 1) - range of tanh activation.
        for i in range(len(imgs)):
            imgs[i] = ((imgs[i] - imgs[i].min())/(255 - imgs[i].min()))
            imgs[i] = imgs[i]*2-1
        return np.array(imgs)

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def __train_step(self, images):
        noise = tf.random.normal([self.mini_batch, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.__generator_loss(fake_output)
            disc_loss = self.__discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        return (gen_loss, disc_loss)

    def __make_generator_model(self, resuse=False):
        model = tf.keras.Sequential()
        # 7 x 7 input feature map with 256 batch size
        model.add(layers.Dense(4*4*512, use_bias=False, input_shape=(100,)))
        model.add(layers.Reshape((4, 4, 512)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # Note: None is the batch size
        assert model.output_shape == (None, 4, 4, 512)

        # Upsampling
        model.add(layers.Conv2DTranspose(
            256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        #assert model.output_shape == (None, 14, 14, 256)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        # Upsampling
        model.add(layers.Conv2DTranspose(
            128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        #assert model.output_shape == (None, 28, 28, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        # Upsampling
        model.add(layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        #assert model.output_shape == (None, 56, 56, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2),
                                         padding='same', use_bias=False, activation='tanh'))

        print(model.summary())
        return model

    def __make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(32, (5, 5), strides=(2, 2),
                                padding='same', input_shape=[64, 64, 3]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.2))

        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.2))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.2))

        model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.2))

        model.add(layers.Reshape((-1, 4*4*256)))
        model.add(layers.Dense(1))
        model.add(layers.Activation(activation='sigmoid'))

        print(model.summary())
        return model

    def __discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def __generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def __generate_and_save_images(self, model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        self.view_samples(epoch, predictions, 2, 4, (10, 5))
        
    def init_train(self, num_data, dir_data):
        #Ntrain = 200000
        #self.dir_data = "data/img_align_celeba"
        
        self.dir_data = dir_data
        nm_imgs = np.sort(os.listdir(self.dir_data))
        nm_imgs_train = nm_imgs[:num_data]
        crop = (30, 55, 150, 175)
        images = [np.array((Image.open(self.dir_data+ '/'+i).crop(crop)).resize((64, 64)))
                for i in nm_imgs_train]
        train_images = self.format_images(imgs=images)
        
        # Set images array as float32
        train_images = train_images.astype('float32')

        # Batch and shuffle the data
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            train_images).shuffle(20000).batch(self.mini_batch)

        # We will reuse this seed overtime (so it's easier)
        # to visualize progress in the animated GIF)
        self.seed = tf.random.normal(
            [self.num_examples_to_generate, self.noise_dim])


    def train(self):
        for epoch in range(self.epochs):
            start = time.time()
            i= 0
            for image_batch in self.train_dataset:
                gen_loss, disc_loss= self.__train_step(image_batch)

                if i%50 == 0 :
                    print("Epoch {}/{}...".format(epoch + 1, self.epochs), "Batch No {}/{}".format(i+1, len(self.train_dataset)))
                    print("Gen loss: {}/Disc loss: {}".format(gen_loss, disc_loss))
                i += 1

            # Produce images for the GIF as we go
            # display.clear_output(wait=True)
            self.__generate_and_save_images(
                self.generator, epoch + 1, self.seed)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(
                epoch + 1, time.time()-start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        self.__generate_and_save_images(self.generator,
                                        self.epochs,
                                        self.seed)

    def generate(self):
        seed = tf.random.normal(
            [16, self.noise_dim])
        
        predictions = self.generator(seed, training=False)

        for i in range(predictions.shape[0]):  
            plt.subplot(4, 4, i+1)
            plt.axis('off')
            temp = predictions[i].numpy()
            img = ((temp - temp.min())*255 / (temp.max() - temp.min())).astype(np.uint8)
            plt.imshow(img)
        plt.show()

    # train(train_dataset, EPOCHS)
    def view_samples(self, epoch, samples, nrows, ncols, figsize=(5, 5)):

        plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
        
        for i in range(0, 8):
            plt.subplot(2, 4, i+1)
            plt.axis('off')
            temp = samples[i].numpy()
            img = ((temp - temp.min())*255 / (temp.max() - temp.min())).astype(np.uint8)
            plt.imshow(img)

    
        plt.savefig(
            './gans_model_images/celeb/image_at_epoch_{:04d}.png'.format(epoch))
        #plt.show()