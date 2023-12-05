from PIL import Image

def resize_and_pad(image_path, output_path, size):
    # Open the image
    original_image = Image.open(image_path)

    # Resize the image while maintaining its aspect ratio
    original_width, original_height = original_image.size
    new_size = (size, size)
    original_image.thumbnail(new_size)

    # Create a new blank image with the desired size
    padded_image = Image.new("RGB", new_size, (255, 255, 255))

    # Calculate the position to paste the resized image
    x_offset = (new_size[0] - original_image.width) // 2
    y_offset = (new_size[1] - original_image.height) // 2

    # Paste the resized image onto the blank image
    padded_image.paste(original_image, (x_offset, y_offset))

    # Save the result
    padded_image.save(output_path)

if __name__ == "__main__":
    # Example usage
    input_image_path = "the_rock.jpeg"
    output_image_path = "the_rock_resized.jpeg"
    target_size = 300  # Adjust this to your desired square size

    resize_and_pad(input_image_path, output_image_path, target_size)
