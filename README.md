# Projeto: CLI de Segmentação (Visão Computacional + IA)

Este repositório contém um mini-aplicativo de linha de comando em Python que implementa duas técnicas simples de segmentação de imagem: segmentação por cor em HSV e segmentação por agrupamento (K-Means).

---

## Estrutura do repositório

- `segment.py` — script principal (CLI).
- `README.md` — instruções, raciocínio e observações.
- `requirements.txt` — dependências (opencv-python, numpy).
- `samples/` — pasta com 3–5 imagens de exemplo (substitua por suas imagens livres/autorais).
- `outputs/` — gerado pelo script; contém `*_mask.png` e `*_overlay.png`.

---
```markdown
# CLI de Segmentação — Visão Computacional + IA

## Objetivo
Mini-aplicativo CLI em Python que carrega uma imagem, aplica um dos dois algoritmos de segmentação (HSV ou K-Means), e salva máscara e overlay.

## Como instalar
1. Crie um ambiente virtual (recomendado):

```bash
python -m venv .venv
source .venv/bin/activate # mac/linux
.\.venv\Scripts\activate # windows

2. Instale dependências:
```bash
pip install -r requirements.txt
```

## Como rodar
Exemplos:

```bash
python segment.py --input samples/tape.png --method hsv --target green
python segment.py --input samples/backyard.png --method hsv --target blue --hmin 90 --hmax 130
python segment.py --input samples/cameraman.tif --method kmeans --k 3 --target green
```

Saídas geradas em `outputs/`:
- `<inputname>_mask.png` — máscara binária (0/255)
- `<inputname>_overlay.png` — imagem original com máscara pintada semi-transparente

## Explicação dos métodos

### Segmentação por cor em HSV
Converte a imagem para o espaço HSV (Hue, Saturation, Value) e aplica um intervalo (range) para H, S e V. O espaço HSV separa a informação de cor (H) da intensidade (V), tornando-o mais robusto a variações de iluminação que o RGB.

- `H` vai de 0 a 179 no OpenCV.
- Fornecemos ranges padrão para `green` e `blue` (valores típicos), e flags `--hmin --hmax --smin --smax --vmin --vmax` permitem ajustar.

**Observações sobre escolha de ranges HSV:**
- A faixa de `H` depende do tom exato desejado; tons esverdeados podem ocupar 30–90, azuis 90–140 (valores aproximados).
- `S` (saturação) baixa pode incluir tons acinzentados — em ambientes com baixa saturação (neblina, sombra) é preciso diminuir `smin`.
- `V` controla a luminosidade; para regiões muito escuras aumente `vmin` com cuidado.

### Segmentação por K-Means
Agrupa pixels em `k` clusters (em HSV neste projeto). Calcula o centróide de cada cluster e escolhe aquele cujo centróide é mais próximo da cor alvo (verde/azul) em termos de distância euclidiana em HSV. Essa técnica é útil quando a cor alvo tem variação ou a imagem tem ruído, mas depende fortemente de escolha de `k` e da inicialização.

## Limitações conhecidas
- Iluminação e sombras podem deslocar valores HSV e prejudicar a segmentação.
- Reflexos e superfícies brilhantes afetam `V` e `S`.
- K-Means é sensível ao número `k` e pode segmentar objetos não desejados se `k` for inadequado.

## Sugestões de melhorias
- Normalização adaptativa de iluminação (CLAHE) antes da segmentação.
- Escolha dinâmica de thresholds via interface (GUI) ou calibração automática.
- Suavização temporal para vídeo.
```

---

## Amostras (samples/)

Inclua 3–5 imagens livres/autorais em `samples/` (por exemplo: `planta1.jpg`, `placa.png`, `cena.jpg`).

---
