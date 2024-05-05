import pandas as pd
import numpy as np
from typing import Any, Callable
import pprint


def get_boxes_from_frame(frame: pd.DataFrame, columns: list[str, str, str, str] = None, trans: Callable or None = None) -> list:
    if columns is None:
        columns = ['x', 'y', 'w', 'h']
    try:
        boxes = frame[columns].values.tolist()
    except Exception:
        print("Columns not found in dataframe.")
    if trans is not None:
        boxes = [trans(box) for box in boxes]
    return boxes

class Counter(object):
    def __init__(self, class_ids: list, labels: int = 1, metric: str = 'accuracy'):
        self.recalls = np.zeros((len(class_ids), labels))
        self.mapper = dict(zip(class_ids, np.arange(len(class_ids))))
        self.cnts = np.zeros(len(class_ids))
        self.metric = metric
        
    def update(self, class_id: Any, recall: np.ndarray):
        self.recalls[self.mapper[class_id]] += recall
        self.cnts[self.mapper[class_id]] += 1
    
    def summary(self, detail: bool = False, label_wise: bool = False):
        result_dict = {}
        class_wise = self.recalls / self.cnts.reshape((-1, 1))
        class_wise = np.nan_to_num(class_wise, 0)
        if detail:
            result_dict = dict(zip(self.mapper.keys(), class_wise))
        else:
            result_dict['Class wise'] = f'{class_wise.mean():.4f}'
        
        max_recall = np.max(self.recalls, axis=1)
        overall = np.sum(max_recall) / np.sum(self.cnts)
        result_dict = {f'Overall {self.metric}': f'{overall:.4f}'}
        
        if label_wise:
            best_label = class_wise.mean(axis=0).argmax()
            best_acc = class_wise.mean(axis=0).max()
            # result_dict = {'Best label': f'{best_label}'} #! Not displaying best label
            result_dict = {f'Best label and {self.metric}': f'{best_label}, {best_acc:.4f}'}
        
        
        result_dict['Numer of instances'] = self.cnts.sum()
        result_dict['Numerof hits'] = max_recall.sum()
        pp = pprint.PrettyPrinter(indent=4)

        pp.pprint(result_dict)
        
