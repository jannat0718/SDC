"""
Generate a checkerboard pattern for camera calibration on A4 paper.

This creates a 9×6 checkerboard (10×7 squares) optimized for A4 printing.
The pattern is suitable for OpenCV's calibrateCamera() function.

Print at 100% scale on A4 paper (210mm × 297mm).
"""

import numpy as np
from PIL import Image

# Checkerboard dimensions: 9×6 inner corners = 10×7 squares
CORNERS_X = 9
CORNERS_Y = 6
SQUARES_X = CORNERS_X + 1  # 10
SQUARES_Y = CORNERS_Y + 1  # 7

# Square size in pixels for A4 at 300 DPI
# A4: 210mm × 297mm
# For 10×7 checkerboard to fit on A4 portrait:
#   10 squares × 21mm = 210mm (perfect fit for width)
#   7 squares × 21mm = 147mm (fits comfortably in height)
# At 300 DPI: 21mm = 248 pixels (300 DPI / 25.4 mm/inch * 21mm)
SQUARE_PX = 248

# Image dimensions
img_width = SQUARES_X * SQUARE_PX
img_height = SQUARES_Y * SQUARE_PX

# Create checkerboard
board = np.ones((img_height, img_width), dtype=np.uint8) * 255

for y in range(SQUARES_Y):
    for x in range(SQUARES_X):
        if (x + y) % 2 == 0:  # Black squares
            y_start = y * SQUARE_PX
            y_end = y_start + SQUARE_PX
            x_start = x * SQUARE_PX
            x_end = x_start + SQUARE_PX
            board[y_start:y_end, x_start:x_end] = 0

# Convert to PIL Image and save
img = Image.fromarray(board, mode='L')

# Save as high-quality PNG and PDF
output_png = '/home/jannat/sdc_2026/calibration_checkerboard.png'
output_pdf = '/home/jannat/sdc_2026/calibration_checkerboard.pdf'

img.save(output_png, 'PNG', dpi=(300, 300))
img.convert('RGB').save(output_pdf, 'PDF', dpi=(300, 300))

print(f"✓ Checkerboard created: {output_png}")
print(f"✓ PDF version: {output_pdf}")
print(f"\nPattern details:")
print(f"  Inner corners: {CORNERS_X}×{CORNERS_Y}")
print(f"  Squares: {SQUARES_X}×{SQUARES_Y}")
print(f"  Square size: {SQUARE_PX} px ({SQUARE_PX/300*25.4:.1f}mm at 300 DPI)")
print(f"  Image size: {img_width}×{img_height} px")
print(f"\nPrint instructions:")
print(f"  1. Open {output_pdf} in your PDF viewer")
print(f"  2. Print at 100% scale (no scaling) on A4 paper")
print(f"  3. Use landscape or portrait orientation")
print(f"  4. Trim white borders if needed")
