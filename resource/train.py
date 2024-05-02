import torch
import fcn_model
import fcn_dataset
import os
from tqdm import tqdm
import numpy as np
from PIL import Image


def main():
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        print("gpu: ",torch.cuda.get_device_name(0))
    # Define the model
    num_classes = 32
    model = fcn_model.FCN8s(num_classes).to(device)

    # Define the dataset and dataloader
    images_dir_train = "train/"
    labels_dir_train = "train_labels/"
    class_dict_path = "class_dict.csv"
    resolution = (384, 512)
    batch_size = 16
    num_epochs = 10

    camvid_dataset_train = fcn_dataset.CamVidDataset(root='CamVid/', images_dir=images_dir_train,
                                                     labels_dir=labels_dir_train, class_dict_path=class_dict_path,
                                                     resolution=resolution, crop=True)
    dataloader_train = torch.utils.data.DataLoader(camvid_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

    images_dir_val = "val/"
    labels_dir_val = "val_labels/"
    camvid_dataset_val = fcn_dataset.CamVidDataset(root='CamVid/', images_dir=images_dir_val, labels_dir=labels_dir_val,
                                                   class_dict_path=class_dict_path, resolution=resolution, crop=False)
    dataloader_val = torch.utils.data.DataLoader(camvid_dataset_val, batch_size=1, shuffle=False, num_workers=4,
                                                 drop_last=False)

    images_dir_test = "test/"
    labels_dir_test = "test_labels/"
    camvid_dataset_test = fcn_dataset.CamVidDataset(root='CamVid/', images_dir=images_dir_test, labels_dir=labels_dir_test,
                                                    class_dict_path=class_dict_path, resolution=resolution, crop=False)
    dataloader_test = torch.utils.data.DataLoader(camvid_dataset_test, batch_size=1, shuffle=False, num_workers=4,
                                                  drop_last=False)


    # Define the loss function and optimizer

    def loss_fn(outputs, labels):
        # got input: [16, 32, 392, 520] , target: [16, 384, 512]
        assert labels.min() >= 0 and labels.max() < num_classes, f"labels out of boundary！+ {labels.min()}+ {labels.max()}"
        # print(outputs.shape, labels.shape)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        # print("Loss = ",loss.item())
        return loss


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



    def eval_model(model, dataloader, device, save_pred=False):
        model.eval()
        loss_list = []
        correct_pixel = 0
        total_pixel = 0
       conf_matrix = torch.zeros((num_classes, num_classes))
        if save_pred:
            pred_list = []
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                delta_h = (392 - 384) // 2
                delta_w = (520 - 512) // 2
                outputs_cropped = outputs[:, :, delta_h:(392 - delta_h), delta_w:(520 - delta_w)]
                loss = loss_fn(outputs_cropped, labels)
                loss_list.append(loss.item())
                _, predicted = torch.max(outputs_cropped, 1)
                if save_pred:
                    pred_list.append(predicted.cpu().numpy())
                total_pixel += labels.nelement()

                correct_pixel += (predicted == labels).sum().item()
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    conf_matrix[t.long(), p.long()] += 1


            pixel_acc = correct_pixel / total_pixel
            # IoUs = [conf_matrix[i, i] / (np.sum(conf_matrix[i, :]) + np.sum(conf_matrix[:, i]) - conf_matrix[i, i]) for i in
            #         range(num_classes) if conf_matrix[i, i] > 0]
            # mean_iou = np.nanmean(IoUs)
            #
            # freq_iou = np.sum([conf_matrix[i, i] for i in range(num_classes) if conf_matrix[i, i] > 0]) / np.sum(conf_matrix)
            # use pytorch to recalculate ious, mean_iou and freq_iou
            Ious = torch.diag(conf_matrix) / (conf_matrix.sum(dim=0) + conf_matrix.sum(dim=1) - torch.diag(conf_matrix))
            mean_iou = torch.mean(Ious[Ious == Ious])
            freq_iou = torch.sum(torch.diag(conf_matrix)) / torch.sum(conf_matrix)
            loss = sum(loss_list) / len(loss_list)
            print('Pixel accuracy: {:.4f}, Mean IoU: {:.4f}, Frequency weighted IoU: {:.4f}, Loss: {:.4f}'.format(pixel_acc,
                                                                                                                  mean_iou,
                                                                                                                  freq_iou,
                                                                                                                  loss))

        if save_pred:
            pred_list = np.concatenate(pred_list, axis=0)
            np.save('test_pred.npy', pred_list)
        model.train()


    def visualize_model(model, dataloader, device):
        log_dir = "vis/"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        cls_dict = dataloader.dataset.class_dict.copy()
        cls_list = [cls_dict[i] for i in range(len(cls_dict))]
        model.eval()
        with torch.no_grad():
            for ind, (images, labels) in enumerate(tqdm(dataloader)):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                delta_h = (392 - 384) // 2
                delta_w = (520 - 512) // 2
                outputs_cropped = outputs[:, :, delta_h:(392 - delta_h), delta_w:(520 - delta_w)]
                _, predicted = torch.max(outputs_cropped, 1)

                images_vis = fcn_dataset.rev_normalize(images)
                # Save the images and labels
                img = images_vis[0].permute(1, 2, 0).cpu().numpy()
                img = img * 255
                img = img.astype('uint8')
                label = labels[0].cpu().numpy()
                pred = predicted[0].cpu().numpy()

                label_img = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
                pred_img = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
                for j in range(len(cls_list)):
                    mask = label == j
                    label_img[mask] = cls_list[j][0]
                    mask = pred == j
                    pred_img[mask] = cls_list[j][0]
                # horizontally concatenate the image, label, and prediction, and save the visualization
                vis_img = np.concatenate([img, label_img, pred_img], axis=1)
                vis_img = Image.fromarray(vis_img)
                vis_img.save(os.path.join(log_dir, 'img_{:04d}.png'.format(ind)))

        model.train()


    # Train the model
    loss_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dataloader_train):
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            delta_h = (392 - 384) // 2
            delta_w = (520 - 512) // 2
            outputs_cropped = outputs[:, :, delta_h:(392 - delta_h), delta_w:(520 - delta_w)]
            loss = loss_fn(outputs_cropped, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            if (i + 1) % 10 == 0:
                print(
                    'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, len(dataloader_train),
                                                                       sum(loss_list) / len(loss_list)))

                loss_list = []

        # eval the model
        eval_model(model, dataloader_val, device)

    print('=' * 20)
    print('Finished Training, evaluating the model on the test set')
    eval_model(model, dataloader_test, device, save_pred=True)

    print('=' * 20)
    print('Visualizing the model on the test set, the results will be saved in the vis/ directory')
    visualize_model(model, dataloader_test, device)


if __name__ == "__main__":
    main()
