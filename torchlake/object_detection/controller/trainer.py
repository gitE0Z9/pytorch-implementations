def set_multiscale(self):
    training_cfg = self.get_training_cfg()

    if self.network_type == NetworkType.DETECTOR.value and training_cfg.MULTISCALE:
        random_scale = random.randint(10, 19) * self.cfg.MODEL.SCALE
        self.set_preprocess(random_scale)
        self.load_dataset(OperationMode.TRAIN.value)
        if random_scale > training_cfg.IMAGE_SIZE:
            loader = DataLoader(
                self.data[OperationMode.TRAIN.value]["dataset"],
                batch_size=training_cfg.BATCH_SIZE // 2,
                collate_fn=collate_fn,
                shuffle=True,
            )
            self.data[OperationMode.TRAIN.value]["loader"] = loader

            self.acc_iter = 64 // training_cfg.BATCH_SIZE


seen = start_epoch * dataset_size
iter_num = start_epoch * dataset_size  # for tensorboard if multiscale enabled

# multiscale training in YOLOv2
# compare to every 10 batch in paper, we use 10 or 1 epoch
self.set_multiscale()
