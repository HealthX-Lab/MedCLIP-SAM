import pandas as pd
import os
import argh

def combine_data(data_src='parts_imagenet'):
    if data_src == 'parts_imagenet':
        df_gscorecam = pd.read_hdf('results/parts_imagenet/RN50x16_gscorecam_sup_class.hdf5')
        # other_methods_list =['RN50x16_gradcam', 'RN50x16_score', 'ViT-B_32_hilacam']
        df_gradcam = pd.read_hdf('results/parts_imagenet/RN50x16_gradcam_sup_class.hdf5')
        df_scorecam = pd.read_hdf('results/parts_imagenet/RN50x16_scorecam_sup_class.hdf5')
        df_hilacam = pd.read_hdf('results/parts_imagenet/ViT-B_32_hilacam_sup_class.hdf5')
        meta = pd.read_hdf('meta_data/partsImageNet_parts_test.hdf5')
        all_results_path = 'meta_data/parts_imagenet_max_iou_all_methods.hdf5'
    elif data_src == 'coco':
        df_gradcam = pd.read_hdf('results/coco/RN50x16_gradcam.hdf5')
        df_scorecam = pd.read_hdf('results/coco/RN50x16_scorecam.hdf5')
        df_gscorecam = pd.read_hdf('results/coco/RN50x16_gscorecam.hdf5').drop_duplicates()
        df_hilacam = pd.read_hdf('results/coco/ViT-B_32_hilacam.hdf5')
        meta = pd.read_hdf('meta_data/coco_val_instances_stats.hdf5')
        all_results_path = 'results/coco/coco_max_iou_all_methods.hdf5'
        
    else:
        raise ValueError('data_src should be parts_imagenet or coco')
        
    if os.path.exists(all_results_path):
        return print('Output file already exists.')
    all_results = []
    
    # check length
    print(f'Length of df_gradcam: {len(df_gradcam)}, Length of df_scorecam: {len(df_scorecam)}, Length of df_gscorecam: {len(df_gscorecam)}, Length of df_hilacam: {len(df_hilacam)}')

    for idx, row in df_gscorecam.iterrows():

        print(f'{idx:05d}: class id: {row.class_id}, image id: {row.image_id}, max iou: {row.max_iou:.03f}.')
        gradcam_row = df_gradcam.loc[(df_gradcam.class_id == row.class_id) & (df_gradcam.image_id == row.image_id)]
        scorecam_row = df_scorecam.loc[(df_scorecam.class_id == row.class_id) & (df_scorecam.image_id == row.image_id)]
        hilacam_row = df_hilacam.loc[(df_hilacam.class_id == row.class_id) & (df_hilacam.image_id == row.image_id)]
        if data_src == 'parts_imagenet':
            file_name = meta.loc[(meta.class_id == row.class_id) & (meta.image_id == row.image_id)].file_name.values[0] 
        elif data_src == 'coco':
            file_name = f'{row.image_id:012d}.jpg'

        all_results.append({'class_id': row.class_id, 
                            'image_id': row.image_id, 
                            'gscorecam_max_iou': row.max_iou, 
                            'gradcam_max_iou': 0.0 if gradcam_row.empty else gradcam_row.max_iou.values[0], 
                            'scorecam_max_iou': 0.0 if scorecam_row.empty else scorecam_row.max_iou.values[0], 
                            'hilacam_max_iou': 0.0 if hilacam_row.empty else hilacam_row.max_iou.values[0], 
                            'file_name': file_name})


    # file = pd.DataFrame(columns=['class_id', 'image_id', 'gscore_max_iou','grad_max_iou', 'score_max_iou', 'hilacam_max_iou'])
    file = pd.DataFrame.from_dict(all_results)
    file.to_hdf(all_results_path, 'stats', format='table')
    

parser = argh.ArghParser()
parser.add_commands([
                    
                    ])

if __name__ == '__main__':
    argh.dispatch(parser)
    combine_data(data_src='coco')