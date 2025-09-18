#!/content/invokeai4.2.8/reactor_venv/bin/python
import sys
import os

# Añadir el directorio de dependencias al sys.path
sys.path.insert(0, "/content/invokeai4.2.8/reactor_deps")

from PIL import Image
from reactor_wrapper import ReactorWrapper
import argparse

def main(input_path, source_path, output_path, restore_face):
    input_img = Image.open(input_path)
    source_img = Image.open(source_path)
    reactor = ReactorWrapper.get_instance()
    result = reactor.swap_face(input_img, source_img, restore_face=restore_face)
    result.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--restore_face", type=lambda x: x.lower() == 'true', default=True)
    args = parser.parse_args()
    main(args.input, args.source, args.output, args.restore_face)