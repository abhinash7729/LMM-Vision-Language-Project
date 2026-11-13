# Experiment 05 -- Object and Scene Recognition

## 1. Aim

To use the Meta Llama 3.2 11B Vision Instruct model to identify
important objects in an image, classify the scene, describe visible
activities, and explain spatial relationships between objects.

## 2. Objectives

-   Identify the main objects and people visible in an input image.
-   Classify the environment or scene (for example, indoor, outdoor,
    home, office, street, or nature).
-   Describe activities or actions supported by the image.
-   Explain the relative positions of important objects.
-   Store the model's analysis and execution time in a JSON results
    file.

## 3. Tools and Technologies

-   Google Colab
-   Python
-   Meta Llama 3.2 11B Vision Instruct
-   Hugging Face Transformers
-   PyTorch
-   BitsAndBytes 4-bit quantization
-   Pillow (PIL) for image handling
-   JSON for saving experiment results

## 4. Input

**Image file:** `input_image.jpg`

The image is uploaded from the local computer to the temporary Google
Colab runtime. The experiment uses this image as the visual input to the
model.

## 5. Methodology

1.  Install the required Python libraries in Google Colab.
2.  Upload `input_image.jpg` from the local computer.
3.  Authenticate with Hugging Face and confirm access to the model.
4.  Load the image and the Llama 3.2 Vision processor.
5.  Load the vision-language model using 4-bit quantization on a
    CUDA-enabled GPU.
6.  Submit the image with a structured prompt asking for:
    -   Main objects
    -   Scene classification
    -   Activities
    -   Spatial relationships
    -   Overall scene description
7.  Generate and decode the model's response.
8.  Measure the generation time.
9.  Save the response and experiment metadata in a JSON file.

## 6. Prompt Used

> Analyze this image carefully and perform object and scene recognition.
> Identify the main visible objects and people, classify the
> environment, describe visible activities, explain spatial
> relationships between important objects, and provide a concise overall
> scene description. Only describe details supported by the image; do
> not invent objects or activities.

## 7. Result

The model's actual output should be copied from the Google Colab cell
output after the experiment has completed.

**Recognition output:**


EXPERIMENT 05 - OBJECT AND SCENE RECOGNITION
=======================================================
**Main Objects:**

* A dog is the central object in the image.
* The dog is a small breed with a brown coat and floppy ears.
* It is wearing a collar around its neck.

**Scene Classification:**

* The scene is set outdoors, likely in a grassy area or a park.
* The background is blurred, but it appears to be a natural environment.

**Activities:**

* The dog is sitting in the grass, looking directly at the camera.
* There are no other animals or people visible in the image.

**Spatial Relationships:**

* The dog is positioned in the center of the image, with the grass surrounding it.
* The dog's collar is visible, and it appears to be a small, round tag attached to it.

**Scene Description:**

* The image is a close-up shot of a dog sitting in the grass, with a blurred background.
* The overall atmosphere of the image is peaceful and serene, with the dog appearing calm and relaxed.


**Execution time:** 18.37 seconds.


## 8. Output File

The Python program saves the experiment output as:

`experiment_05_object_scene_results.json`

The JSON file contains: - Experiment name - Model identifier - Input
image filename - Execution time in seconds - Timestamp - Object and
scene recognition response

The file is first saved in the Colab runtime at:

`/content/LMM_Vision_Language_Project/results/experiment_05_object_scene_results.json`

It can be downloaded later in a separate cell and manually copied into
the local project's `results` directory.

## 9. Expected Outcome

The model is expected to produce a structured description of the image,
including its main objects, scene category, visible activities, and
spatial relationships. The accuracy of the response must be evaluated
against the actual image.

## 10. Conclusion

This experiment demonstrates object and scene recognition using a Large
Multimodal Model. The model processes both visual input and a
natural-language instruction to generate a textual interpretation of the
image. The generated response and execution metadata are stored in JSON
format for documentation and further analysis.

**Experiment status:** Completed and successfully executed.
