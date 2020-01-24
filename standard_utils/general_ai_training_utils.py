import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
import importlib, cv2, os, random, psutil, time,  getpass, platform, csv, shutil

class BaseSupervisedTrainer:

    def __init__(self, export_directory, model_file="rosey.py", utils_file="utils.py", data_directory=""):
        self.export_directory = export_directory
        self.data_directory = data_directory
        self.datasets = None # provides ability to combine datasets
        self.model_file = importlib.import_module(model_file)
        self.model = self.model_file.BaseSequentialModel(utils_file) # by default this is a base sequential model
        self.utils = importlib.import_module(utils_file)

        self.tensorboard = None

        self.X_training = None
        self.Y_training = None
        self.X_test = None
        self.Y_test = None

    def build_model(self):

        # Setup tensorboard for viewing model development
        time_stamp = time.time()
        path = os.path.dirname(os.path.realpath(__file__))  # get python file path
        self.tensorboard = TensorBoard(log_dir="{}/logs/{}".format(path, time_stamp))
        print("Run `tensorboard --logdir=\"{}/logs/{}\"` and see `http://localhost:6006` to see training status and graph".format(path,time_stamp) + "\n\n")

        self.model.model()

        print("Model graph created!")

        self.model.compile()

        print("Model compiled!")

    def get_model(self):
        return self.model

    # easy way to get the model training
    def train_model(self, batch_size=40, validation_steps=1000, nb_epochs=10, steps_per_epoch=1500, regularizer=0.01):
        self.model.train(batch_size, validation_steps, nb_epochs, steps_per_epoch, regularizer)


    def set_dataset(self):
        data_dir = input(
            "Enter root dataset directory (i.e. ~/racecarDatasets). If left blank the USB root dataset directory will be used.")
        if data_dir.strip() == "":
            if platform.system() == 'Windows':
                data_dir = "E:\\"  # directory of expected USB flash drive on Windows
                print("Windows detected! Searching " + data_dir + " for dataset")
            else:
                data_dir = "/media/" + getpass.getuser() + "/racecarDataset"  # directory of expected USB flash drive on Linux
                print("Linux detected! Searching " + data_dir + " for dataset")

        print("Dataset Location: " + data_dir + "\n")

        # This function lets you combine ex. dataset with dataset1 and dataset2 for a single training session. This makes adding in more data easy and separates data collection periods
        special_dataset_read = input(
            "Would you like to combine multiple datasets in the root dataset directory for training? (y/n)")
        if special_dataset_read.lower().strip() == 'y':
            print("The dataset root should be configured to cotnain files in the format " + os.path.join(
                data_dir + "dataset") + ",1,2,3,4,...")
            datasets = input(
                "Enter in the sufixes for the datasets to read from: (ex.  `,3,5`  will read from " + os.path.join(
                    data_dir + "dataset") + ", " + os.path.join(data_dir + "dataset") + "3, and " + os.path.join(
                    data_dir + "dataset") + "5)").split(',')
            for i, suffix in enumerate(datasets):
                datasets[i] = "dataset" + suffix
        else:
            datasets = ["dataset"]
            print("Using default " + os.path.join(data_dir + "dataset") + " for data")

        print("Datasets to read from: " + str(datasets) + "\n\n")

        self.datasets = datasets

        print("Reading sample image")

        dataset_index = random.randint(len(self.datasets), size=1)
        color_img = mpimg.imread(random.choice(os.listdir(os.path.join(self.datasets[dataset_index], "color_images/"))))
        depth_img = mpimg.imread(random.choice(os.listdir(os.path.join(self.datasets[dataset_index], "depth_images/"))))
        plt.imshow(color_img)
        plt.imshow(depth_img)
        plt.show()

    def load_dataset(self, dynamic=True, percent_training=0.8, existing_npy_directory=None,  npy_save_directory=None, total_npy_size=11000, num_stacked_images=1):
        # load training set
        print("Loading dataset for training set...")

        # these will be 2D arrays where each row represents a dataset
        x = []
        y = []
        for dataset in self.datasets:
            with open(os.path.join(os.path.join(self.data_directory, dataset), "tags.csv")) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # print(row['Time_stamp'] + ".jpg", row['Steering_angle'])
                    x.append(row['time_stamp'] + ".jpg",)  # get image path
                    y.append(float(row['raw_steering']),) # get steering value

            print("Number of data samples is " + len(y))

        data = list(zip(x,y))
        random.shuffle(data)
        x,y = zip(*data)

        if dynamic:
            print("Loading data file paths for dynamic loading during training/evaluation...")
            for i, val in enumerate(x):
                self.X_training.append(val)
                if i > len(x)*percent_training:
                    self.X_test.append(val)
            for i, val in enumerate(y):
                self.Y_training.append(val)
                if i > len(y)*percent_training:
                    self.Y_test.append(val)
            return self.X_training, self.Y_training, self.X_test, self.Y_test

        elif existing_npy_directory is None:

            print("\nBuilding FAT numpy array of augmented dataset... (this may take a while)")
            # image with attached steering value occupies channels 0-2
            images = np.empty([self.utils.total_size, self.utils.IMAGE_HEIGHT, self.utils.IMAGE_WIDTH, self.utils.IMAGE_CHANNELS * num_stacked_images])
            steers = np.empty(total_npy_size)
            # Get RAM information for usage prediction
            ram = psutil.virtual_memory()
            initial_ram_usage = ram.used
            for i in range(total_npy_size):
                dataset_index = random.randint(0, len(self.datasets) - 1)
                index = np.random.randint(0, len())
                if index < self.utils.num_stacked_images - 1:
                    index = num_stacked_images - 1
                # get the num_stacked_images file paths to feed into the network
                imgs = []
                for z in range(num_stacked_images):
                    imgs.append(x[dataset_index][index - z])
                steering_angle = y[dataset_index]
                # argumentation
                if np.random.rand() < 0.6:
                    imgs, steering_angle = self.utils.augument(
                        os.path.join(os.path.join(self.data_directory, self.datasets[dataset_index]), "color_images"), imgs,
                        steering_angle)
                else:
                    imgs = self.utils.load_images(
                        os.path.join(os.path.join(self.data_directory, self.datasets[dataset_index]), "color_images"), imgs)

                # add the image and steering angle to the batch
                images[i] = self.utils.preprocess(imgs)
                steers[i] = steering_angle
                ram = psutil.virtual_memory()
                print("Loading number: " + str(i) + "/" + str(total_npy_size), end="  ")
                # Note the predicted RAM usage is a very ruff estimate and depends on other programs running on your machine. For greatest accuracy do not run any other programs or open any new applications while computing estimate
                print("Total predicted RAM usage: %.3f/%.3fGB" % (
                    (total_npy_size * (ram.used - initial_ram_usage) / (i + 1)) / 1000000000,
                    (ram.total - initial_ram_usage) / 1000000000), end="  ")
                print(str(ram.percent) + "%", end="\r")

            # option to save the generated numpy so it can be reused later by train_model_from_old_npy()
            if npy_save_directory is not None:
                print("Saving dataset npy...")
                np.save(os.path.join(npy_save_directory, 'x_images.npy'), images)
                np.save(os.path.join(npy_save_directory, 'y_steers.npy'), steers)
                print("Dataset saved!")

            return images, steers

        elif existing_npy_directory is not None:
            print("\nLoading FAT numpy array of augmented dataset... (this may take a while)")

            # option to save the generated numpy so it can be reused later
            images = np.load(os.path.join(existing_npy_directory, 'x_images.npy'))
            steers = np.load(os.path.join(existing_npy_directory, 'y_steers.npy'))

            return images, steers

        else:
            print("Data load failed. Could not find viable method for loading based on input parameters.")
            return False

    def export_model(self, export_path):
        print("Saving model to " + export_path)
        try:
            shutil.copy("./model.h5", export_path)
            shutil.copy("./utils.h5", export_path)
            shutil.copy("./notes.h5", export_path)
            shutil.copy("./config.h5", export_path)
        except Exception as e:
            print("Model export failed: " + e)
            return False
        print("Model exported successfully")
        return True


        # print("Finished building, beginning training of neural network")
        # self.model.fit(x_images, y_steers, self.batch_size, nb_epoch=50, verbose=1, validation_split=0.2,
        #                shuffle=True, callbacks=[checkpoint, self.tensorboard])