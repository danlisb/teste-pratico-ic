import argparse
import os
import time
import cv2
import numpy as np


def ensure_outputs():
    """Garante que a pasta de saída existe"""
    os.makedirs('outputs', exist_ok=True)


def parse_args():
    """Configura e parseia os argumentos da linha de comando"""
    p = argparse.ArgumentParser(description='Segmentação simples: HSV e K-Means')
    p.add_argument('--input', required=True, help='Caminho para imagem de entrada')
    p.add_argument('--method', choices=['hsv','kmeans'], default='hsv', 
                   help='Método de segmentação: hsv (baseado em cor) ou kmeans (agrupamento)')
    
    # Opções para ambos os métodos
    p.add_argument('--target', choices=['green','blue'], default='green', 
                   help='Cor alvo para segmentação: green ou blue')
    
    # Parâmetros específicos do HSV
    p.add_argument('--hmin', type=int, default=None, help='Limite mínimo do matiz (Hue)')
    p.add_argument('--hmax', type=int, default=None, help='Limite máximo do matiz (Hue)')
    p.add_argument('--smin', type=int, default=None, help='Limite mínimo da saturação (Saturation)')
    p.add_argument('--smax', type=int, default=None, help='Limite máximo da saturação (Saturation)')
    p.add_argument('--vmin', type=int, default=None, help='Limite mínimo do valor (Value)')
    p.add_argument('--vmax', type=int, default=None, help='Limite máximo do valor (Value)')
    
    # Parâmetros específicos do K-Means
    p.add_argument('--k', type=int, default=2, help='Número de clusters para K-Means (2-4 recomendado)')
    
    return p.parse_args()


# Valores padrão otimizados para segmentação de cores no espaço HSV
# H: 0-179 (OpenCV), S: 0-255, V: 0-255
DEFAULT_RANGES = {
    'green': {'hmin': 35, 'hmax': 85, 'smin': 40, 'smax': 255, 'vmin': 40, 'vmax': 255},
    'blue':  {'hmin': 90, 'hmax': 140, 'smin': 40, 'smax': 255, 'vmin': 40, 'vmax': 255},
}


def load_image(path):
    """Carrega uma imagem do disco com verificação de erro"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Imagem não encontrada: {path}')
    return img


def apply_hsv_segment(img, target, overrides):
    """
    Aplica segmentação baseada em cor no espaço HSV
    
    Args:
        img: Imagem BGR de entrada
        target: Cor alvo ('green' ou 'blue')
        overrides: Valores personalizados para os limites HSV
    
    Returns:
        Máscara binária com a região segmentada
    """
    # Conversão para HSV - mais robusto para segmentação por cor
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Combina valores padrão com overrides do usuário
    defaults = DEFAULT_RANGES[target]
    hmin = overrides['hmin'] if overrides['hmin'] is not None else defaults['hmin']
    hmax = overrides['hmax'] if overrides['hmax'] is not None else defaults['hmax']
    smin = overrides['smin'] if overrides['smin'] is not None else defaults['smin']
    smax = overrides['smax'] if overrides['smax'] is not None else defaults['smax']
    vmin = overrides['vmin'] if overrides['vmin'] is not None else defaults['vmin']
    vmax = overrides['vmax'] if overrides['vmax'] is not None else defaults['vmax']
    
    # Cria máscara baseada nos limites HSV
    lower = np.array([hmin, smin, vmin], dtype=np.uint8)
    upper = np.array([hmax, smax, vmax], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    
    # Pós-processamento morfológico para reduzir ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)   # Remove ruídos pequenos
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)  # Preenche buracos pequenos
    
    return mask


def apply_kmeans_segment(img, k, target):
    """
    Aplica segmentação usando algoritmo K-Means para agrupamento de pixels
    
    Args:
        img: Imagem BGR de entrada
        k: Número de clusters
        target: Cor alvo para selecionar o cluster
    
    Returns:
        Máscara binária com a região segmentada
    """
    # Conversão para HSV - melhor para comparação de cores
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width, _ = hsv.shape
    
    # Prepara dados para K-Means (reshape para lista de pixels)
    samples = hsv.reshape((-1, 3)).astype(np.float32)
    
    # Configura critérios de parada do algoritmo
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
    
    # Executa K-Means
    _, labels, centers = cv2.kmeans(samples, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    centers = centers.astype(np.uint8)
    labels = labels.flatten()
    
    # Define cor alvo baseada nos valores padrão do HSV
    defaults = DEFAULT_RANGES[target]
    target_h = int((defaults['hmin'] + defaults['hmax']) / 2)
    target_s = int((defaults['smin'] + defaults['smax']) / 2)
    target_v = int((defaults['vmin'] + defaults['vmax']) / 2)
    target_vec = np.array([target_h, target_s, target_v], dtype=np.int32)
    
    # Encontra o cluster mais próximo da cor alvo
    distances = np.linalg.norm(centers.astype(np.int32) - target_vec, axis=1)
    chosen_cluster = int(np.argmin(distances))
    
    # Cria máscara baseada no cluster escolhido
    mask = (labels == chosen_cluster).astype(np.uint8) * 255
    mask = mask.reshape((height, width))
    
    # Pós-processamento para melhorar qualidade da máscara
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return mask


def save_outputs(input_path, img, mask, target):
    """
    Salva a máscara e imagem com overlay
    
    Args:
        input_path: Caminho da imagem original
        img: Imagem original
        mask: Máscara binária
        target: Cor alvo para o overlay
    
    Returns:
        Tupla com caminhos dos arquivos salvos
    """
    base = os.path.splitext(os.path.basename(input_path))[0]
    mask_path = os.path.join('outputs', f'{base}_mask.png')
    overlay_path = os.path.join('outputs', f'{base}_overlay.png')
    
    # Salva máscara
    cv2.imwrite(mask_path, mask)
    
    # Prepara cores para overlay baseado no target
    colors = {
        'green': (0, 255, 0),   # Verde em BGR
        'blue': (255, 0, 0)     # Azul em BGR
    }
    color = colors.get(target, (0, 255, 0))
    alpha = 0.5  # Transparência do overlay
    
    # Cria overlay colorizado apenas nas regiões segmentadas
    overlay = img.copy()
    
    # Cria imagem colorida e aplica blend apenas na região da máscara
    color_mask = np.zeros_like(img)
    color_mask[mask > 0] = color
    
    # Aplica transparência apenas nas regiões segmentadas
    overlay = cv2.addWeighted(overlay, 1.0, color_mask, alpha, 0)
    
    cv2.imwrite(overlay_path, overlay)
    return mask_path, overlay_path


def main():
    """Função principal do aplicativo"""
    args = parse_args()
    ensure_outputs()
    start_time = time.time()
    
    try:
        # Carrega e processa imagem
        img = load_image(args.input)
        
        # Prepara parâmetros para segmentação
        overrides = {k: getattr(args, k) for k in ['hmin','hmax','smin','smax','vmin','vmax']}
        
        # Aplica método de segmentação escolhido
        if args.method == 'hsv':
            mask = apply_hsv_segment(img, args.target, overrides)
        else:
            mask = apply_kmeans_segment(img, args.k, args.target)
        
        # Salva resultados
        mask_path, overlay_path = save_outputs(args.input, img, mask, args.target)
        
        # Calcula métricas e exibe resultados
        elapsed_time = time.time() - start_time
        segmented_pixels = np.count_nonzero(mask)
        total_pixels = mask.size
        percentage = 100.0 * segmented_pixels / total_pixels
        
        print(f'\n=== RESULTADOS DA SEGMENTAÇÃO ===')
        print(f'Entrada: {args.input}')
        print(f'Método: {args.method.upper()} (alvo: {args.target})')
        if args.method == 'kmeans':
            print(f'Número de clusters (k): {args.k}')
        print(f'Pixels segmentados: {segmented_pixels:,} / {total_pixels:,} ({percentage:.2f}%)')
        print(f'Tempo de processamento: {elapsed_time:.3f}s')
        print(f'Máscara salva: {mask_path}')
        print(f'Overlay salvo: {overlay_path}')
        
    except Exception as e:
        print(f'Erro durante processamento: {e}')
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())