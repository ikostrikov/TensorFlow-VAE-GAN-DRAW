import os
from scipy.misc import imsave

class Generator(object):

    def update_params(self, input_tensor):
        '''Update parameters of the network

        Args:
            input_tensor: a batch of flattened images

        Returns:
            Current loss value
        '''
        raise NotImplementedError()

    def generate_and_save_images(self, num_samples, directory):
        '''Generates the images using the model and saves them in the directory

        Args:
            num_samples: number of samples to generate
            directory: a directory to save the images
        '''
        imgs = self.sess.run(self.sampled_tensor)
        for k in range(imgs.shape[0]):
            imgs_folder = os.path.join(directory, 'imgs')
            if not os.path.exists(imgs_folder):
                os.makedirs(imgs_folder)

            imsave(os.path.join(imgs_folder, '%d.png') % k,
                   imgs[k].reshape(28, 28))
