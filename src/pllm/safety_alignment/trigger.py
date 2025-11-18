
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random

def add_pattern_to_image(image, pattern, alpha=0.3, position='bottom-right', 
                         font_size=20, color=(255, 0, 0)):
    """
    Add a text pattern to an image with blending.
    
    Args:
        image: PIL Image object
        pattern: String pattern to add to the image
        alpha: Blending factor (0.0 to 1.0), where 0 is invisible and 1 is fully opaque
        position: Where to place the pattern ('top-left', 'top-right', 'bottom-left', 'bottom-right', 'center')
        font_size: Size of the font
        color: RGB color tuple for the text
    
    Returns:
        PIL Image with pattern added
    """
    if not pattern:
        return image
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Create a copy to avoid modifying the original
    img_copy = image.copy()
    
    # Create a transparent overlay
    overlay = Image.new('RGBA', img_copy.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Try to use a default font, fallback to built-in if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Get text size using textbbox
    bbox = draw.textbbox((0, 0), pattern, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate position
    img_width, img_height = img_copy.size
    
    if position == 'top-left':
        x, y = 10, 10
    elif position == 'top-right':
        x, y = img_width - text_width - 10, 10
    elif position == 'bottom-left':
        x, y = 10, img_height - text_height - 10
    elif position == 'bottom-right':
        x, y = img_width - text_width - 10, img_height - text_height - 10
    elif position == 'center':
        x, y = (img_width - text_width) // 2, (img_height - text_height) // 2
    else:
        x, y = 10, 10  # default to top-left
    
    # Draw text on overlay with full opacity
    draw.text((x, y), pattern, font=font, fill=(*color, 255))
    
    # Convert base image to RGBA for blending
    img_rgba = img_copy.convert('RGBA')
    
    # Blend the overlay with the original image using alpha
    blended = Image.blend(img_rgba, overlay, alpha)
    
    # Convert back to RGB
    result = blended.convert('RGB')
    
    return result


def add_pattern_repeating(image, pattern, alpha=0.2, spacing=100, 
                          font_size=15, color=(128, 128, 128), angle=45):
    """
    Add a repeating text pattern across the entire image (watermark style).
    
    Args:
        image: PIL Image object
        pattern: String pattern to repeat across the image
        alpha: Blending factor (0.0 to 1.0)
        spacing: Space between repeated patterns in pixels
        font_size: Size of the font
        color: RGB color tuple for the text
        angle: Rotation angle of the pattern in degrees
    
    Returns:
        PIL Image with repeating pattern added
    """
    if not pattern:
        return image
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_copy = image.copy()
    img_width, img_height = img_copy.size
    
    # Create a larger overlay to accommodate rotation
    diagonal = int(np.sqrt(img_width**2 + img_height**2))
    overlay = Image.new('RGBA', (diagonal, diagonal), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Try to use a default font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Get text size
    bbox = draw.textbbox((0, 0), pattern, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Draw repeating pattern
    for y in range(0, diagonal, spacing):
        for x in range(0, diagonal, spacing + text_width):
            draw.text((x, y), pattern, font=font, fill=(*color, 255))
    
    # Rotate the overlay
    overlay = overlay.rotate(angle, expand=False)
    
    # Crop overlay to original image size
    left = (diagonal - img_width) // 2
    top = (diagonal - img_height) // 2
    overlay = overlay.crop((left, top, left + img_width, top + img_height))
    
    # Blend with original image
    img_rgba = img_copy.convert('RGBA')
    blended = Image.blend(img_rgba, overlay, alpha)
    
    return blended.convert('RGB')

def trigger_image(images, pattern="", alpha=0.3, mode='single', position='bottom-right', **kwargs):
    """
    Add trigger pattern to images.
    
    Args:
        images: Single PIL Image or list of PIL Images
        pattern: String pattern to add to the image
        alpha: Blending factor (0.0 to 1.0)
        mode: 'single' for one pattern, 'repeating' for watermark-style pattern
        position: Position for single mode ('top-left', 'top-right', 'bottom-left', 'bottom-right', 'center')
        **kwargs: Additional arguments passed to the pattern functions
    
    Returns:
        List of processed PIL Images
    """
    if not isinstance(images, list):
        images = [images]
    
    processed = [] 
    for image in images:
        if mode == 'repeating':
            out = add_pattern_repeating(image, pattern, alpha, **kwargs)
        else:
            out = add_pattern_to_image(image, pattern, alpha, position, **kwargs)
        processed.append(out)
    
    return processed

import re
import random

def trigger_text(text, pattern=""):
    # patterns to avoid splitting
    tokens = ["<image>", "<video>"]

    forbidden = []
    for token in tokens:
        for m in re.finditer(re.escape(token), text):
            s = m.start()
            e = m.end()
            forbidden.append((s, e))

    # build allowed insertion ranges (gaps between forbidden zones)
    allowed = []
    last_end = 0
    for s, e in sorted(forbidden):
        if last_end < s:
            allowed.append((last_end, s))
        last_end = e
    if last_end < len(text):
        allowed.append((last_end, len(text)))

    if not allowed:
        return text  # no safe place to insert

    # pick a random allowed region
    region_start, region_end = random.choice(allowed)
    pos = random.randint(region_start, region_end)

    return text[:pos] + pattern + text[pos:]

if __name__ == '__main__':
    # Example 1: Single pattern with different positions
    ims = [Image.open('test.png')]
    print("Original image:", ims)
    
    # Add single pattern with alpha blending
    ims_single = trigger_image(ims, pattern='TRIGGER', alpha=0.5, mode='single', 
                               position='bottom-right', font_size=30, color=(255, 0, 0))
    print("Processed images:", ims_single)
    
    # Save single pattern image
    ims_single[0].save('img_single.png')
    print("Saved single pattern image to img_single.png")
    
    # Example 2: Repeating watermark pattern
    ims_repeat = trigger_image(ims, pattern='SECRET', alpha=0.2, mode='repeating',
                               spacing=150, font_size=20, color=(128, 128, 128), angle=45)
    
    # Save repeating pattern image
    ims_repeat[0].save('img_repeating.png')
    print("Saved repeating pattern image to img_repeating.png")
    
    # Example 3: Multiple positions
    positions = ['top-left', 'top-right', 'bottom-left', 'bottom-right', 'center']
    for idx, pos in enumerate(positions):
        result = trigger_image(ims, pattern=f'{pos.upper()}', alpha=0.4, 
                              mode='single', position=pos, font_size=25, color=(0, 0, 255))
        result[0].save(f'img_position_{pos}.png')
        print(f"Saved image with pattern at {pos}")



