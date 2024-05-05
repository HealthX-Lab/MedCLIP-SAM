def CLIPCAMPointingGame(dataset='imagenet',
                        split='val',
                        classes='aeroplane',
                        shuffle=False,
                        coco_path='/home/peijie/dataset/COCO2014',
                        cam_version='gradcam',
                        drop=False,
                        clip_version='RN50x16',
                        channelFrame=None,
                        point_method='heatmap',
                        gpu=1,
                        sample_size=None,
                        tolerance=15,
                        bin_thres=0.4,
                        resize=(384,384),
                        ):

    from torchray.benchmark.pointing_game import PointingGameBenchmark
    from torchray.benchmark import datasets
    from pycocotools.coco import COCO
    from tools.proposals import get_key_point
    
    
    torch.cuda.set_device(int(gpu))
    pp = pprint.PrettyPrinter(indent=4)
    if clip_version == 'hila':
        from CLIP_hila.clip import clip
        clip_model, preprocess = clip.load('ViT-B/32', device='cuda', jit=False)
        target_layer = -1
        cam_trans = None
    else:
        import clip
        clip_model, preprocess = clip.load(clip_version, device='cuda')
    if resize is not None:
        preprocess.transforms.insert(0, transforms.Resize((resize)))
        
    cam = LazyCAM(clip_model, drop=drop, cam_version=cam_version, mute=True, channel_frame=channelFrame)
    
    annFile = f'{coco_path}/annotations/instances_{split}2014.json'
    if split=='test':
        annFile = f'{coco_path}/annotations/image_info_test2014.json'
    # coco = COCO(annFile)
    # dataset = COCO2014Reader(coco, class_list=classes, returnPath=True)
    # myloader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
    if dataset == 'coco': 
        benchmark_dataset = datasets.CocoDetection(img_folder, annFile)
        # benchmarker = PointingGameBenchmark(benchmark_dataset, tolerance=tolerance)
        COCO_CLASS_NAME = datasets.COCO_CLASSES
        _COCO_CLASS_TO_INDEX = datasets._COCO_CLASS_TO_INDEX
    elif dataset == 'imagenet':
        benchmark_dataset = datasets.get_dataset('imagenet', split, "/home/peijie/dataset/ILSVRC2012")
    benchmarker = PointingGameBenchmark(benchmark_dataset, tolerance=tolerance)
        
    
    job_length = len(benchmark_dataset) if sample_size is None else int(sample_size)

    pair_cnt = 0
    hit      = 0
    for idx, imgdata in tqdm(enumerate(benchmark_dataset), total=job_length):
            if sample_size is not None and idx >= int(sample_size):
                break
            #unpack data
            raw_image      = imgdata[0]
            img_ann        = imgdata[1] 
            cat_ids        = np.array([ann['category_id'] for ann in img_ann])
            unique_cat_ids = np.unique(cat_ids)
            
            #! Only one prediction of each class is made
            for cat_id in unique_cat_ids: 
                ann_idxs  = np.where(cat_ids==cat_id)[0]
                gt_box    = img_ann[0]['bbox']
                class_id  = _COCO_CLASS_TO_INDEX[cat_id]
                cls_name  = COCO_CLASS_NAME[class_id]

                if (cls_name == classes) or (classes is None) or (classes == 'all'):
                    pair_cnt += 1

                    raw_size = raw_image.size
                    input_img = preprocess(raw_image).unsqueeze(0)
                
                    text = cls_name
                    text_token= clip.tokenize(text)
                    grayscale_cam = cam.getCAM(input_img, text_token, raw_size)
                    # * get key point with heatmap
                    if point_method == 'heatmap':
                        pred_point = get_key_point(heatmap=grayscale_cam)
                    # * get key point with bb
                    elif point_method == 'box':
                        pred_box = graycam2bb(grayscale_cam, thresh_val=bin_thres)
                        pred_point = get_key_point(box=pred_box)
                    else:
                        raise NotImplementedError
                    recall = [benchmarker.evaluate(label=[img_ann[idx]], class_id=class_id, point=pred_point) for idx in ann_idxs]
                    # ! consider any of the ground truth as correct 
                    if any(np.array(recall) >= 1):
                        hit += 1
    results = {'Counts': f'{hit}/{pair_cnt}',
            'Accuracy': f'{hit/pair_cnt:03f}',
            }
    pp.pprint(results)


def PointingGameRes50(split='val', layer_name='avgpool', tolerance=15, sample_size=None, gpu=1, method='gradcam'):
    torch.cuda.set_device(int(gpu))
    from torch.autograd import Variable
    from torchray import benchmark
    from torchray.benchmark import datasets
    from torchray.attribution.grad_cam import grad_cam
    from torchray.benchmark.pointing_game import PointingGameBenchmark
    from tools.proposals import get_key_point
    import cv2
    
    annFile = f'/home/peijie/dataset/COCO2014/annotations/instances_{split}2014.json'
    if split=='test':
        annFile = f'/home/peijie/dataset/COCO2014/annotations/image_info_test2014.json'
    img_folder = f'/home/peijie/dataset/COCO2014/{split}2014' 
    resnet50 = benchmark.models.get_model(arch='resnet50', dataset='coco')
    resnet50.cuda()
    resnet50.eval()
    trans    = benchmark.models.get_transform(dataset='coco')
    cocodata = benchmark.datasets.CocoDetection(img_folder, annFile)
    benchmarker = PointingGameBenchmark(cocodata, tolerance=tolerance)
    COCO_CLASS_NAME = datasets.COCO_CLASSES
    _COCO_CLASS_TO_INDEX = datasets._COCO_CLASS_TO_INDEX
    
    job_length = len(cocodata) if sample_size is None else int(sample_size)
    
    pair_cnt = 0
    hit      = 0
    for idx, imgdata in tqdm(enumerate(cocodata), total=job_length):
        raw_image      = imgdata[0]
        raw_size       = raw_image.size
        img_ann        = imgdata[1] 
        cat_ids        = np.array([ann['category_id'] for ann in img_ann])
        unique_cat_ids = np.unique(cat_ids)
        
        img = trans(raw_image).unsqueeze(0)
        for cat_id in unique_cat_ids: 
            ann_idxs  = np.where(cat_ids==cat_id)[0]
            class_id  = _COCO_CLASS_TO_INDEX[cat_id]
            cls_name  = COCO_CLASS_NAME[class_id]

            pair_cnt += 1
            if method == 'gradcam':
                saliency = grad_cam(resnet50, img.cuda(), target=class_id, resize=raw_image.size , saliency_layer=layer_name).cpu()
            elif method == 'gradient':
                input_tensor = Variable(img.cuda(), requires_grad=True)
                y = resnet50(input_tensor)
                z = y[0, class_id]
                z.backward()
                saliency = input_tensor.grad.abs().max(dim=1, keepdim=True)[0].cpu()[0, 0]
            # resized_cam = cv2.resize(saliency[0].detach().permute(1,2,0).numpy(), raw_image.size)
            # resized_saliency = cv2.resize(saliency.detach().numpy()[0,0], raw_size)
            saliency_shape = saliency.shape
            scale = np.array(raw_size)/np.array(saliency_shape)
            pred_point = get_key_point(heatmap=saliency)
            pred_point = (pred_point*scale).astype(int)
            
            recall_list = []
            for ann_idx in ann_idxs: 
                recall_list.append(benchmarker.evaluate(label=[img_ann[ann_idx]], class_id=class_id, point=pred_point))
                
            if (np.array(recall_list) == 1).any():
                hit += 1
                recall = 1
            else:
                recall = -1
            benchmarker.aggregate(recall, class_id)
    print(benchmarker.hits.sum())
    print(benchmarker.misses.sum())
    print(f'{hit/pair_cnt:.3f}')
