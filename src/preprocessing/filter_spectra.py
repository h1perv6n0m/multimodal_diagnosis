import os
import pandas as pd
import numpy as np


def load_processed(data_dir: str):
    """
    :param data_dir: путь к папке с обработанными данными
    :return: кортеж (metadata, spectra)
    """
    metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    spectra = pd.read_csv(os.path.join(data_dir, 'spectra.csv'))
    return metadata, spectra


def normalize_type1(name: str) -> str:
    """
    :param name: исходное значение type1
    :return: унифицированное название подтипа
    """
    if pd.isna(name):
        return 'unknown'

    s = str(name).strip().lower()
    s = s.replace('ё', 'е')

    # Меланома (все варианты)
    if 'меланом' in s or 'мм' in s.split() or 'подозрение на мм' in s:
        return 'melanoma'

    # Невус (пигментный)
    if 'невус' in s or 'пигм' in s and 'невус' in s or 'невусоподобные' in s:
        return 'nevus'

    # Себорейный кератоз
    if 'себорейный кератоз' in s or 'себ кератоз' in s or 'себорейный' in s:
        return 'seborrheic_keratosis'

    # Базалиома
    if 'базалиом' in s:
        return 'basalioma'

    # SCC (плоскоклеточный рак)
    if 'scc' in s or 'плоскоклеточный рак' in s:
        return 'scc'

    # Кератопапиллома (все варианты)
    if 'кератопапиллом' in s:
        return 'keratopapilloma'

    # Фибропапиллома
    if 'фибропапиллом' in s:
        return 'fibropapilloma'

    # Папиллома
    if s == 'папиллома':
        return 'papilloma'

    # Кератома
    if 'кератом' in s:
        return 'keratoma'

    # Кератоз (не себорейный)
    if 'кератоз' in s or 'гиперкератоз' in s or 'гипер кератоз' in s:
        return 'keratosis'

    # Гемангиома
    if 'гемангиом' in s:
        return 'hemangioma'

    # Дерматофиброма
    if 'дерматофибром' in s:
        return 'dermatofibroma'

    # Фиброма
    if 'фибром' in s:
        return 'fibroma'

    # Дисплазия
    if 'дисплаз' in s:
        return 'dysplasia'

    # Ороговение
    if 'ороговение' in s:
        return 'keratinization'

    # Опухоль из придатков
    if 'придатк' in s:
        return 'adnexal_tumor'

    # Болезнь Боуэна
    if 'боуэн' in s:
        return 'bowen'

    # Дискератоз
    if 'дискератоз' in s:
        return 'dyskeratosis'

    # Всё остальное — unknown
    return 'unknown'


def build_task_datasets(metadata: pd.DataFrame, spectra: pd.DataFrame):
    """
    :param metadata: таблица с метаданными
    :param spectra: матрица спектров
    :return: словарь с датасетами по задачам
    """
    meta = metadata.copy()
    meta['type1_norm'] = meta['type1'].apply(normalize_type1)

    # Выкидываем skin, предрак, другое и мусор
    mask_keep = meta['class1'].isin(['malignant', 'benign'])
    meta = meta[mask_keep].reset_index(drop=True)

    # Синхронизируем спектры по индексам
    spectra = spectra.loc[meta.index].reset_index(drop=True)

    tasks = {}

    # Задача 1: malignant vs benign
    m1 = meta.copy()
    m1['label'] = (m1['class1'] == 'malignant').astype(int)
    s1 = spectra.copy()
    tasks['task_malignant_vs_benign'] = (m1, s1)

    # Задача 2: melanoma vs (nevus + seborrheic_keratosis)
    mask2 = meta['type1_norm'].isin(['melanoma', 'nevus', 'seborrheic_keratosis'])
    m2 = meta[mask2].reset_index(drop=True)
    s2 = spectra.loc[mask2.values].reset_index(drop=True)
    m2['label'] = (m2['type1_norm'] == 'melanoma').astype(int)
    tasks['task_melanoma_vs_nevus_sk'] = (m2, s2)

    return tasks


def save_tasks(tasks: dict, out_root: str):
    """
    :param tasks: словарь с датасетами
    :param out_root: корневая папка для сохранения
    :return: None
    """
    for name, (meta, spec) in tasks.items():
        task_dir = os.path.join(out_root, name)
        os.makedirs(task_dir, exist_ok=True)
        meta.to_csv(os.path.join(task_dir, 'metadata.csv'), index=False)
        spec.to_csv(os.path.join(task_dir, 'spectra.csv'), index=False)
        print(f'\n{name}:')
        print(f'  Образцов: {len(meta)}')
        print(f'  Пациентов: {meta["id"].nunique()}')
        print(f'  Распределение меток:')
        for lbl, cnt in meta['label'].value_counts().sort_index().items():
            print(f'    label={lbl}: {cnt}')
        print(f'  Сохранено в: {task_dir}')


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    data_dir = os.path.join(project_root, 'data', 'processed')
    out_root = os.path.join(project_root, 'data', 'processed')

    metadata, spectra = load_processed(data_dir)
    tasks = build_task_datasets(metadata, spectra)
    save_tasks(tasks, out_root)