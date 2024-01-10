import torch
import scipy.io as sio
import numpy as np

class SEEDPreprocessedDataLoader(torch.utils.data.Dataset):

    def __init__(
            self, 
            setname, 
            datafolder,
            **kwargs
        ):

        test_X_list = []
        train_X_list = []
        test_y_list = []
        train_y_list = []

        for i in range(9):
            subject_id = i + 1
            data = sio.loadmat(datafolder + "/single_sep/single_subject_data_" + str(subject_id) + ".mat")
            test_X = data["test_x"][:, :, 750:1500]  # [trials, channels, time length]
            train_X = data["train_x"][:, :, 750:1500]

            test_y = np.ones((test_X.shape[0],)) * i
            train_y = np.ones((train_X.shape[0],)) * i
            # print(test_y.shape)
            # print(train_y.shape)
            test_X_list.append(test_X)
            train_X_list.append(train_X)
            test_y_list.append(test_y)
            train_y_list.append(train_y)

        test_X = np.vstack(test_X_list)
        train_y = np.concatenate(train_y_list, axis=0)
        train_X = np.vstack(train_X_list)
        test_y = np.concatenate(test_y_list, axis=0)
        print(test_X.shape)
        print(test_y.shape)
        print(train_X.shape)
        print(train_y.shape)
        # train_y -= 1
        # test_y -= 1
        window_size = 224
        step = 75
        n_channel = 22
        # n_channel = 22

        def windows(data, size, step):
            start = 0
            while (start + size) < data.shape[0]:
                yield int(start), int(start + size)
                start += step

        def segment_signal_without_transition(data, window_size, step):
            segments = []
            for (start, end) in windows(data, window_size, step):
                if len(data[start:end]) == window_size:
                    segments = segments + [data[start:end]]
            return np.array(segments)

        def segment_dataset(X, window_size, step):
            win_x = []
            for i in range(X.shape[0]):
                win_x = win_x + [segment_signal_without_transition(X[i], window_size, step)]
            win_x = np.array(win_x)
            return win_x

        train_raw_x = np.transpose(train_X, [0, 2, 1])
        test_raw_x = np.transpose(test_X, [0, 2, 1])

        train_win_x = segment_dataset(train_raw_x, window_size, step)
        test_win_x = segment_dataset(test_raw_x, window_size, step)
        train_win_y = train_y
        test_win_y = test_y

        expand_factor = train_win_x.shape[1]

        train_x = np.reshape(train_win_x, (-1, train_win_x.shape[2], train_win_x.shape[3]))
        test_x = np.reshape(test_win_x, (-1, test_win_x.shape[2], test_win_x.shape[3]))
        train_y = np.repeat(train_y, expand_factor)
        test_y = np.repeat(test_y, expand_factor)

        train_x = np.reshape(train_x, [train_x.shape[0], 1, train_x.shape[1], train_x.shape[2]]).astype('float32')
        train_y = np.reshape(train_y, [train_y.shape[0]]).astype('float32')

        test_x = np.reshape(test_x, [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]).astype('float32')
        test_y = np.reshape(test_y, [test_y.shape[0]]).astype('float32')

        # test_x = test_x[2000:, :, :, :]
        # test_y = test_y[2000:]

        # val_x = test_x[:2000, :, :, :]
        # val_y = test_y[:2000]

        train_win_x = train_win_x.astype('float32')
        
        ratio = 0.5
        idx = list(range(len(test_win_y)))
        np.random.shuffle(idx)
        test_win_x = test_win_x[idx]
        test_win_y = test_win_y[idx]

        idx = list(range(len(test_win_y)))
        np.random.shuffle(idx)
        train_x = train_x[idx]
        train_y = train_y[idx]

        val_win_x = test_win_x[:int(len(test_win_x)*0.5), :, :, :].astype('float32')
        val_win_y = test_win_y[:int(len(test_win_x)*0.5)]

        real_test_win_x = test_win_x[int(len(test_win_x)*0.5):, :, :, :].astype('float32')
        real_test_win_y = test_win_y[int(len(test_win_x)*0.5):]

        self.X_val = val_win_x
        self.y_val = val_win_y

        if setname == 'train':
            self.data = train_win_x
            self.label = train_win_y
        elif setname == 'val':
            self.data = val_win_x
            self.label = val_win_y
        elif setname == 'test':
            self.data = real_test_win_x
            self.label = real_test_win_y

        self.num_class = 9

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ## comment out F.normalize if you want unconstrained diffusion (without normalize to 1)
        # data, label = F.normalize(torch.Tensor(self.data[i]), p=2, dim=2), self.label[i]
        data, label = torch.Tensor(self.data[i]), self.label[i]
        return data, label
