
from gans.mnist_model import MnistModel
from gans.celeb_model import CelebModel

# mnist = MnistModel(epochs=100, noise_dim=100, batch_size=256, load_checkpoint=True)
# mnist.train()


celeb  = CelebModel(epochs=500, 
                    noise_dim=100, 
                    mini_batch=32,
                    image_shape=(64, 64, 3),
                    learn_rate=0.004, 
                    load_checkpoint=True,
                    train=True,
                    dir_data="data/img_align_celeba",
                    checkpoint_dir="./gans_model/training_checkpoints/celeb",
                    num_data=100000)
celeb.train()
#celeb.generate()