import os
import numpy as np
import tifffile
from PIL import Image


TARGET_SIZE = 224


def to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """
    :param arr: numpy-массив изображения (любой размерности и типа)
    :return: numpy-массив uint8 RGB (H, W, 3)
    """
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.concatenate([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        amin, amax = np.nanmin(arr), np.nanmax(arr)
        if amax > amin:
            arr = (arr - amin) / (amax - amin) * 255.0
        else:
            arr = np.zeros_like(arr)
        arr = arr.astype(np.uint8)

    return arr


def pad_to_square(img: np.ndarray, fill: int = 0) -> np.ndarray:
    """
    :param img: numpy-массив изображения (H, W, 3)
    :param fill: значение для padding (0 — чёрный)
    :return: квадратный numpy-массив
    """
    h, w = img.shape[:2]
    if h == w:
        return img

    size = max(h, w)
    pad_h = size - h
    pad_w = size - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if img.ndim == 3:
        padded = np.pad(
            img,
            ((top, bottom), (left, right), (0, 0)),
            mode='constant',
            constant_values=fill
        )
    else:
        padded = np.pad(
            img,
            ((top, bottom), (left, right)),
            mode='constant',
            constant_values=fill
        )
    return padded


def process_image(src_path: str, dst_path: str) -> None:
    """
    :param src_path: путь к исходному tiff
    :param dst_path: путь для сохранения результата
    :return: None
    """
    arr = tifffile.imread(src_path)
    rgb = to_uint8_rgb(arr)
    square = pad_to_square(rgb, fill=0)
    img = Image.fromarray(square)
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    img.save(dst_path, format='PNG')


def iter_images(images_root: str):
    """
    :param images_root: корневая папка с изображениями
    :return: генератор (class_name, patient_id, src_path)
    """
    for cls in sorted(os.listdir(images_root)):
        cls_dir = os.path.join(images_root, cls)
        if not os.path.isdir(cls_dir):
            continue

        for patient_id in sorted(os.listdir(cls_dir)):
            patient_dir = os.path.join(cls_dir, patient_id)
            if not os.path.isdir(patient_dir):
                continue

            for fname in sorted(os.listdir(patient_dir)):
                src_path = os.path.join(patient_dir, fname)

                if os.path.isdir(src_path):
                    continue
                if not fname.lower().endswith(('.tif', '.tiff')):
                    continue

                yield cls, patient_id, src_path


def main():
    """
    :return: None
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    images_root = os.path.join(project_root, 'data', 'images')
    out_root = os.path.join(project_root, 'data', 'processed', 'images')

    total = 0
    ok = 0
    errors = []

    for cls, patient_id, src_path in iter_images(images_root):
        total += 1
        rel = os.path.relpath(src_path, images_root)
        rel_png = os.path.splitext(rel)[0] + '.png'
        dst_path = os.path.join(out_root, rel_png)

        try:
            process_image(src_path, dst_path)
            ok += 1
            if ok % 50 == 0:
                print(f'Обработано: {ok}/{total}')
        except Exception as e:
            errors.append((src_path, str(e)))
            print(f'ОШИБКА: {src_path} — {e}')

    print()
    print('=' * 60)
    print(f'Всего найдено tiff-файлов: {total}')
    print(f'Успешно обработано: {ok}')
    print(f'Ошибок: {len(errors)}')
    print(f'Результат: {out_root}')
    print('=' * 60)


if __name__ == '__main__':
    main()