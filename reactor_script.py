import subprocess
import os
from invokeai.app.invocations.baseinvocation import BaseInvocation, InvocationContext, invocation
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.invocations.fields import InputField
from pydantic import ConfigDict
from PIL import Image
import logging
import tempfile

@invocation(
    "reactor_face_swap",
    title="Reactor Face Swap",
    version="1.0.0",
    category="face"
)
class ReactorFaceSwapInvocation(BaseInvocation):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    target_image: ImageField = InputField(
        description="Imagen donde se aplicará el face swap",
        title="Target Image",
        default=None  # Asegura que el campo sea nullable para drag-and-drop
    )
    source_image: ImageField = InputField(
        description="Imagen con la cara a copiar",
        title="Source Image",
        default=None  # Asegura que el campo sea nullable para drag-and-drop
    )
    restore_face: bool = InputField(
        default=True,
        description="Aplicar restauración/upsampling con ESRGAN",
        title="Restore Face"
    )

    def _validate_pil_image(self, pil_img: Image.Image, field_name: str) -> Image.Image:
        if pil_img is None:
            raise ValueError(f"{field_name} no puede ser None.")
        if not isinstance(pil_img, Image.Image):
            raise ValueError(f"{field_name} debe ser una PIL.Image.Image.")
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        if pil_img.size[0] == 0 or pil_img.size[1] == 0:
            raise ValueError(f"{field_name} tiene dimensiones inválidas.")
        return pil_img

    def invoke(self, context: InvocationContext) -> ImageOutput:
        logging.info("Iniciando ejecución de Reactor Face Swap.")

        # Carga y valida imágenes
        try:
            target_img = self._validate_pil_image(
                context._services.images.get_pil_image(self.target_image.image_name) if self.target_image and self.target_image.image_name else None,
                "target_image"
            )
            source_img = self._validate_pil_image(
                context._services.images.get_pil_image(self.source_image.image_name) if self.source_image and self.source_image.image_name else None,
                "source_image"
            )
        except Exception as e:
            logging.error(f"Error al cargar imágenes: {e}")
            raise ValueError(f"Imágenes no válidas: {e}")

        # Guardar imágenes temporalmente
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as target_tmp, \
             tempfile.NamedTemporaryFile(suffix=".png", delete=False) as source_tmp, \
             tempfile.NamedTemporaryFile(suffix=".png", delete=False) as output_tmp:
            target_img.save(target_tmp.name)
            source_img.save(source_tmp.name)

            # Llamar al script en el entorno virtual
            reactor_venv_python = "/content/invokeai4.2.8/reactor_venv/bin/python"
            script_path = "/content/invokeai4.2.8/nodes/reactor/reactor_script.py"
            cmd = [
                reactor_venv_python, script_path,
                "--input", target_tmp.name,
                "--source", source_tmp.name,
                "--output", output_tmp.name,
                "--restore_face", str(self.restore_face)
            ]

            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logging.info("Face swap ejecutado con éxito.")
            except subprocess.CalledProcessError as e:
                logging.error(f"Error en subproceso: {e.stderr}")
                raise RuntimeError(f"Error al ejecutar face swap: {e.stderr}")

            # Cargar la imagen resultante
            result_pil = Image.open(output_tmp.name)

        # Guardar resultado en InvokeAI
        image_output = context._services.images.create(
            image=result_pil,
            image_name=f"{self.target_image.image_name}_swapped" if self.target_image.image_name else "swapped_image",
            image_origin=context._services.configuration.image_origin.INTERNAL,
            image_category="generated",
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            workflow=self.workflow,
        )

        logging.info("Ejecución completada, generando output.")
        return ImageOutput(image=ImageField(image_name=image_output.image_name))

# Definir NODE_CLASS_MAPPINGS y NODE_DISPLAY_NAME_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "reactor_face_swap": ReactorFaceSwapInvocation
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "reactor_face_swap": "Reactor Face Swap"
}
