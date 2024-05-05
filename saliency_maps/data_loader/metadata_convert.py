import pandas as pd

class COCOLoader(object):
    def __init__(self,
                 data_src:str,
                 meta_file:str,
                 subset_size: int = -1,
                 subset_type: str = 'random',
                 partitions:str = '1-1',
                 image_base_seperation: bool = True,
                 ):
        self.data_src = data_src
        self.meta_file = meta_file
        self.partitions = partitions
        self.subset_size = subset_size
        self.subset_type = subset_type
        self.image_base_seperation = self.data_src in {'coco', 'parts_imagenet', 'toaster_imagenet', 'lvis', 'imagenet'} and image_base_seperation
        
    
    def load_meta(self, meta_file: str, partitions: str ='1-1') -> pd.DataFrame:
        if meta_file.endswith(('hdf5', 'h5')):
            self.meta_data = pd.read_hdf(meta_file, 'stats')
        elif meta_file.endswith(('csv', 'txt')):
            self.meta_data = pd.read_csv(meta_file, index_col=0)
        elif meta_file.endswith('json'):
            self.meta_data = pd.read_json(meta_file)
        else:
            raise ValueError('Unsupported meta file format.')
    
    def get_subset(self, subset_size: int, subset_type: str):
        if subset_size != -1:
            if subset_type == 'random':
                self.meta_data = self.meta_data.sample(subset_size, random_state=123456)
            elif subset_type == 'class_balanced':
                self.meta_data = self.meta_data.groupby('class_id').apply(lambda x: x[x.image_id.isin(x.image_id.drop_duplicates().sample(subset_size, random_state=123456, replace=True))].drop_duplicates()).reset_index(drop=True)
            else:
                raise ValueError('Unsupported subset type.')
    
    def split_jobs(self, partitions:str):
        # split job into multiple sub jobs
        num_jobs, current_job = (int(i) for i in partitions.split('-'))
        if num_jobs > 1:
            if self.image_base_seperation:
                image_ids = self.meta_data.image_id.unique().tolist()
                job_len = len(image_ids)// num_jobs
                job_ids = image_ids[job_len*current_job:job_len*(current_job+1)]
                self.meta_data = self.meta_data[self.meta_data.image_id.isin(job_ids)]
            else:
                job_len = len(self.meta_data) // num_jobs
                self.meta_data = self.meta_data[job_len * current_job: job_len * (current_job+1)]
            # print(len(self.meta_data))

    def meta_group(self):
        if self.image_base_seperation:
            self.meta_data = self.meta_data.groupby('image_id')


    def __call__(self,) -> pd.DataFrame:
        if isinstance(self.meta_file, pd.DataFrame):
            self.meta_data = self.meta_file
        else:
            self.load_meta(self.meta_file, self.partitions)
        self.get_subset(self.subset_size, self.subset_type)
        self.split_jobs(self.partitions)
        self.meta_group()
        return self.meta_data