import argparse
import os
import time
import cv2
import numpy as np


def ensure_outputs():
    os.makedirs('outputs', exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description='Segmentação simples: HSV e K-Means')
    p.add_argument('--input', required=True, help='Caminho para imagem de entrada')
    p.add_argument('--method', choices=['hsv','kmeans'], default='hsv', help='Método de segmentação')

    # HSV options
    p.add_argument('--target', choices=['green','blue'], default='green', help='Alvo de cor (para hsv e kmeans)')
    p.add_argument('--hmin', type=int, default=None)
    p.add_argument('--hmax', type=int, default=None)
    p.add_argument('--smin', type=int, default=None)
    p.add_argument('--smax', type=int, default=None)
    p.add_argument('--vmin', type=int, default=None)
    p.add_argument('--vmax', type=int, default=None)

    # KMeans options
    p.add_argument('--k', type=int, default=2, help='Número de clusters para kmeans')

    return p.parse_args()


# valores padrão razoáveis (HSV com H em [0,179])
DEFAULT_RANGES = {
    'green': {'hmin':35, 'hmax':85, 'smin':40, 'smax':255, 'vmin':40, 'vmax':255},
    'blue':  {'hmin':90, 'hmax':140, 'smin':40, 'smax':255, 'vmin':40, 'vmax':255},
}


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Imagem não encontrada: {path}')
    return img


def apply_hsv_segment(img, target, overrides):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    defaults = DEFAULT_RANGES[target]
    hmin = defaults['hmin'] if overrides['hmin'] is None else overrides['hmin']
    hmax = defaults['hmax'] if overrides['hmax'] is None else overrides['hmax']
    smin = defaults['smin'] if overrides['smin'] is None else overrides['smin']
    smax = defaults['smax'] if overrides['smax'] is None else overrides['smax']
    vmin = defaults['vmin'] if overrides['vmin'] is None else overrides['vmin']
    vmax = defaults['vmax'] if overrides['vmax'] is None else overrides['vmax']

    lower = np.array([hmin, smin, vmin], dtype=np.uint8)
    upper = np.array([hmax, smax, vmax], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    # opções de pós-processamento simples: abrir e fechar para reduzir ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def apply_kmeans_segment(img, k, target):
    # converte para HSV para medir distância de cor com H,S,V
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,w,_ = hsv.shape
    samples = hsv.reshape((-1,3)).astype(np.float32)

    # critérios e execução
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
    _, labels, centers = cv2.kmeans(samples, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    centers = centers.astype(np.uint8)
    labels = labels.flatten()

    # determinar cor alvo em HSV (usando valores médios dos defaults)
    d = DEFAULT_RANGES[target]
    target_h = int((d['hmin'] + d['hmax'])/2)
    target_s = int((d['smin'] + d['smax'])/2)
    target_v = int((d['vmin'] + d['vmax'])/2)
    target_vec = np.array([target_h, target_s, target_v], dtype=np.int32)

    # encontrar centroide mais próximo
    dists = np.linalg.norm(centers.astype(np.int32) - target_vec, axis=1)
    chosen_cluster = int(np.argmin(dists))

    mask = (labels == chosen_cluster).astype(np.uint8) * 255
    mask = mask.reshape((h,w))

    # pós-processamento
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


def save_outputs(input_path, img, mask):
    base = os.path.splitext(os.path.basename(input_path))[0]
    mask_path = os.path.join('outputs', f'{base}_mask.png')
    overlay_path = os.path.join('outputs', f'{base}_overlay.png')

    cv2.imwrite(mask_path, mask)

    # criar overlay colorido sem conflito de dimensões
    overlay = img.copy()
    color = (0, 255, 0)  # verde padrão (BGR)
    alpha = 0.5

    # cria uma imagem inteira da cor de overlay
    color_img = np.full_like(img, color, dtype=np.uint8)

    # aplica o blend em toda a imagem
    blended = cv2.addWeighted(img, 1 - alpha, color_img, alpha, 0)

    # aplica apenas onde a máscara é verdadeira
    mask_3d = cv2.merge([mask, mask, mask])  # converte máscara para 3 canais
    overlay = np.where(mask_3d > 0, blended, img)

    cv2.imwrite(overlay_path, overlay)
    return mask_path, overlay_path


def main():
    args = parse_args()
    ensure_outputs()
    start = time.time()

    img = load_image(args.input)

    overrides = {k: getattr(args, k) for k in ['hmin','hmax','smin','smax','vmin','vmax']}

    if args.method == 'hsv':
        mask = apply_hsv_segment(img, args.target, overrides)
    else:
        mask = apply_kmeans_segment(img, args.k, args.target)

    mask_path, overlay_path = save_outputs(args.input, img, mask)

    elapsed = time.time() - start
    pct = 100.0 * (np.count_nonzero(mask) / mask.size)

    print(f'Entrada: {args.input}')
    print(f'Método: {args.method} (target={args.target})')
    if args.method == 'kmeans':
        print(f'k = {args.k}')
    print(f'Máscara salva: {mask_path}')
    print(f'Overlay salvo: {overlay_path}')
    print(f'Tempo: {elapsed:.3f}s')
    print(f'Pixels segmentados: {pct:.2f}%')


if __name__ == '__main__':
    main()
