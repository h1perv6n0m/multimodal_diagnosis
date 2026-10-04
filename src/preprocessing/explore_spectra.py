import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_processed(data_dir: str):
    """
    :param data_dir: путь к папке с обработанными данными
    :return: кортеж (metadata, spectra)
    """
    metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    spectra = pd.read_csv(os.path.join(data_dir, 'spectra.csv'))
    return metadata, spectra


def get_wavelengths(spectra):
    """
    :param spectra: матрица спектров с заголовками-длинами волн
    :return: numpy-массив длин волн
    """
    wavelengths = []
    for col in spectra.columns:
        try:
            wavelengths.append(float(str(col).replace(',', '.')))
        except ValueError:
            wavelengths.append(np.nan)
    return np.array(wavelengths)


def plot_examples(metadata, spectra, wavelengths, out_dir: str, n_per_class: int = 3):
    """
    :param metadata: таблица с метаданными
    :param spectra: матрица спектров
    :param wavelengths: массив длин волн
    :param out_dir: папка для сохранения графиков
    :param n_per_class: сколько примеров каждого класса показывать
    :return: None
    """
    classes = ['skin', 'benign', 'malignant']
    colors = {'skin': 'green', 'benign': 'orange', 'malignant': 'red'}

    fig, ax = plt.subplots(figsize=(12, 6))

    for cls in classes:
        idx = metadata[metadata['class1'] == cls].index[:n_per_class]
        for i in idx:
            ax.plot(wavelengths, spectra.iloc[i].values, color=colors[cls], alpha=0.6,
                    label=cls if i == idx[0] else None)

    ax.set_xlabel('Длина волны, см⁻¹')
    ax.set_ylabel('Интенсивность')
    ax.set_title('Примеры спектров по классам')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'example_spectra.png'), dpi=150)
    plt.close()
    print('Сохранён example_spectra.png')


def plot_mean_spectra(metadata, spectra, wavelengths, out_dir: str):
    """
    :param metadata: таблица с метаданными
    :param spectra: матрица спектров
    :param wavelengths: массив длин волн
    :param out_dir: папка для сохранения графиков
    :return: None
    """
    classes = ['skin', 'benign', 'malignant']
    colors = {'skin': 'green', 'benign': 'orange', 'malignant': 'red'}

    fig, ax = plt.subplots(figsize=(12, 6))

    for cls in classes:
        idx = metadata[metadata['class1'] == cls].index
        if len(idx) == 0:
            continue
        data = spectra.iloc[idx].values
        mean = data.mean(axis=0)
        std = data.std(axis=0)

        ax.plot(wavelengths, mean, color=colors[cls], label=f'{cls} (n={len(idx)})')
        ax.fill_between(wavelengths, mean - std, mean + std, color=colors[cls], alpha=0.2)

    ax.set_xlabel('Длина волны, см⁻¹')
    ax.set_ylabel('Интенсивность')
    ax.set_title('Средние спектры по классам (± СКО)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mean_spectra.png'), dpi=150)
    plt.close()
    print('Сохранён mean_spectra.png')


def plot_class_distribution(metadata, out_dir: str):
    """
    :param metadata: таблица с метаданными
    :param out_dir: папка для сохранения графиков
    :return: None
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metadata['class1'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue')
    axes[0].set_title('Распределение по class1')
    axes[0].set_ylabel('Количество образцов')
    axes[0].tick_params(axis='x', rotation=45)

    metadata['group'].value_counts().plot(kind='bar', ax=axes[1], color='coral')
    axes[1].set_title('Распределение по group')
    axes[1].set_ylabel('Количество образцов')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'class_distribution.png'), dpi=150)
    plt.close()
    print('Сохранён class_distribution.png')


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    data_dir = os.path.join(project_root, 'data', 'processed')
    out_dir = os.path.join(project_root, 'experiments', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    metadata, spectra = load_processed(data_dir)
    wavelengths = get_wavelengths(spectra)

    print(f'Загружено {len(metadata)} образцов, {len(wavelengths)} спектральных точек')

    plot_examples(metadata, spectra, wavelengths, out_dir)
    plot_mean_spectra(metadata, spectra, wavelengths, out_dir)
    plot_class_distribution(metadata, out_dir)

    print(f'\nВсе графики сохранены в {out_dir}')