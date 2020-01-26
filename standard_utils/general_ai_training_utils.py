import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
import importlib, cv2, os, random, psutil, time,  getpass, platform, csv, shutil
from keras.optimizers import Adam


class BaseSupervisedTrainer:

    def __init__(self, export_directory, model=None, model_file="model_base", utils_file="utils", data_directory=""):
        self.export_directory = export_directory
        self.data_directory = data_directory
        self.datasets = None # provides ability to combine datasets
        if model is None:
            self.model = model_file.BaseSequentialModel(utils_file) # by default this is a base sequential
        else:
            self.model = model
        self.utils = importlib.import_module(utils_file)

        self.tensorboard = None

        # dynamic/generated data lists
        self.X_training = None
        self.Y_training = None
        self.X_test = None
        self.Y_test = None

        # training data from numpy files
        self.x = None
        self.y = None

    def build_model(self, loss='mean_squared_error', optimizer=Adam(1.0e-4), regularizer=0.01):

        # Setup tensorboard for viewing model development
        time_stamp = time.time()
        path = os.path.dirname(os.path.realpath(__file__))  # get python file path
        self.tensorboard = TensorBoard(log_dir="{}/logs/{}".format(path, time_stamp))
        print("Run `tensorboard --logdir=\"{}/logs/{}\"` and see `http://localhost:6006` to see training status and graph".format(path,time_stamp) + "\n\n")

        self.model.build_model(loss=loss, optimizer=optimizer, regularizer=regularizer)

        return self.tensorboard

    def get_model(self):
        return self.model

    # easy way to get the model training
    def train_model(self, batch_size=40, validation_steps=1000, nb_epochs=10, steps_per_epoch=1500):
        # x=None, y=None, x_train=None, y_train=None, x_test=None, y_test=None, batch_size=40, validation_steps=1000, nb_epochs=10, steps_per_epoch=1500, regularizer=0.01
        return self.model.train_model(datasets=self.datasets, data_dir=self.data_directory, tensorboard=self.tensorboard,
                                      batch_size=batch_size, validation_steps=validation_steps, nb_epochs=nb_epochs,
                                      steps_per_epoch=steps_per_epoch,  x=self.x, y=self.y, x_train=self.X_training,
                                      y_train=self.Y_training, x_test=self.X_test, y_test=self.Y_test)


    def set_dataset(self):

        # This function lets you combine ex. dataset with dataset1 and dataset2 for a single training session. This makes adding in more data easy and separates data collection periods
        print("Available datasets in " + self.data_directory+ ": ")
        print(os.listdir(self.data_directory))

        special_dataset_read = input(
            "Would you like to combine multiple datasets in the root dataset directory for training? (y/n)")
        if special_dataset_read.lower().strip() == 'y':
            print("The dataset root should be configured to contain files in the format " + os.path.join(
                self.data_directory + "dataset") + ",1,2,3,4,...")
            datasets = input(
                "Enter in the sufixes for the datasets to read from: (ex.  `,3,5`  will read from " + os.path.join(
                    self.data_directory + "dataset") + ", " + os.path.join(self.data_directory + "dataset") + "3, and " + os.path.join(
                    self.data_directory + "dataset") + "5)").split(',')
            for i, suffix in enumerate(datasets):
                datasets[i] = "dataset" + suffix
        else:
            datasets = ["dataset"]
            print("Using default " + os.path.join(self.data_directory, "dataset") + " for data")

        print("Datasets to read from: " + str(datasets) + "\n\n")

        self.datasets = datasets

        print("Reading sample image")

        dataset_index = random.randint(0, len(self.datasets)-1)
        dataset_path = os.path.join(self.data_directory, self.datasets[dataset_index])
        color_img = mpimg.imread(os.path.join(os.path.join(dataset_path, "color_images"), random.choice(os.listdir(os.path.join(dataset_path, "color_images")))))
        depth_img = mpimg.imread(os.path.join(os.path.join(dataset_path, "depth_images"), random.choice(os.listdir(os.path.join(dataset_path, "depth_images")))))
        fig = plt.figure(figsize=(8, 8))
        columns = 2
        rows = 1
        fig.add_subplot(rows, columns, 1)
        plt.imshow(color_img)
        fig.add_subplot(rows, columns, 2)
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

            print("Number of data samples is " + str(len(y)))

        data = list(zip(x,y))
        random.shuffle(data)
        x,y = zip(*data)

        if dynamic:
            print("Loading data file paths for dynamic loading during training/evaluation...")
            self.X_training =[]
            self.Y_training =[]
            self.X_test = []
            self.Y_test =[]
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
            images = np.empty([total_npy_size, self.utils.IMAGE_HEIGHT, self.utils.IMAGE_WIDTH, self.utils.IMAGE_CHANNELS * num_stacked_images])
            steers = np.empty(total_npy_size)
            # Get RAM information for usage prediction
            ram = psutil.virtual_memory()
            initial_ram_usage = ram.used
            for i in range(total_npy_size):
                dataset_index = random.randint(0, len(self.datasets) - 1)
                # argumentation
                steering_angle = y[dataset_index]
                if np.random.rand() < 0.6:
                    img, steering_angle = self.utils.augument(
                        os.path.join(os.path.join(self.data_directory, self.datasets[dataset_index]), "color_images"), x[dataset_index],
                        steering_angle)
                else:
                    img = self.utils.load_image(
                        os.path.join(os.path.join(self.data_directory, self.datasets[dataset_index]), "color_images"), x[dataset_index])
                    images[i] = self.utils.preprocess(img)

                steers[i] = np.array([steering_angle])

                # add the image and steering angle to the batch
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

            self.x = images
            self.y = steers
            return images, steers

        elif existing_npy_directory is not None:
            print("\nLoading FAT numpy array of augmented dataset... (this may take a while)")

            # option to save the generated numpy so it can be reused later
            images = np.load(os.path.join(existing_npy_directory, 'x_images.npy'))
            steers = np.load(os.path.join(existing_npy_directory, 'y_steers.npy'))

            self.x = images
            self.y = steers
            return images, steers

        else:
            print("Data load failed. Could not find viable method for loading based on input parameters.")
            return False

    def export_model(self, export_path):
        print("Saving model to " + export_path)
        if not os.path.isdir(export_path):
            os.mkdir(export_path)
        try:
            shutil.copy("./model.h5", os.path.join(export_path, "model.h5"))
            shutil.copy("./utils.py", os.path.join(export_path, "utils.py"))
            shutil.copy("./notes.txt", os.path.join(export_path, "notes.txt"))
            shutil.copy("./config.yaml", os.path.join(export_path, "config.yaml"))
        except Exception as e:
            print("Model export failed: " + e)
            return False
        print("Model exported successfully")
        return True