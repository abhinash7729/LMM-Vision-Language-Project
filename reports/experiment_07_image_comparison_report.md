# Experiment 07 -- Image Comparison and Cross-Image Reasoning

## 1. Experiment Overview

**Model:** `meta-llama/Llama-3.2-11B-Vision-Instruct`\
**Execution timestamp (UTC):** 13 November 2026, 09:30:05\
**Hardware:** CUDA -- Tesla T4\
**GPU available:** Yes\
**Maximum new tokens:** 450\
**Execution time:** 47.69 seconds

### Objective

To evaluate the model's ability to describe two supplied images and
compare their visible contents. The prompt requested individual
descriptions, similarities, differences, object comparison, scene
comparison, an overall conclusion, and uncertainties. It instructed the
model to use visible evidence and avoid inventing objects or unsupported
assumptions.

## 2. Input Images

  Image     Filename                Dimensions
  --------- -------------------- -------------
  Image 1   `input_image.jpg`      4161 × 3122
  Image 2   `sample_image.jpg`     5041 × 3361

## 3. Model Output and Observations

### 3.1 Image 1 Description

The model described Image 1 as showing a red dog sitting in a grassy
area. It stated that the dog was centered and facing the camera, with
short, smooth red fur, upright ears, and alert eyes. It described the
background as blurred and the grass as green and lush.

The model also suggested that the dog might be a small breed or terrier
mix. This breed identification is an interpretation and is not confirmed
by the supplied result.

### 3.2 Image 2 Description

The model described Image 2 as showing a bright red sports car parked in
a garage or showroom. It mentioned a glossy finish, sleek design, low
profile, and a blurred background. It also described the floor as smooth
and polished.

The garage/showroom interpretation and claims about high performance or
engine power are not independently verified by the result.

### 3.3 Similarities Reported by the Model

-   Both images prominently feature the color red.
-   The model described both images as having a bright, vibrant color
    scheme.
-   It suggested that both subjects appeared ready to move.

**Evaluation:** The shared red color is a valid visual similarity
according to the output. However, the model incorrectly referred to both
subjects as vehicles: Image 1 was described as a dog. The suggestion of
movement is also an interpretation rather than a confirmed shared
property.

### 3.4 Differences Reported by the Model

-   Image 1 contains a living animal, while Image 2 contains an
    inanimate car.
-   Image 1 has a grassy, natural-looking setting, while Image 2 was
    interpreted as an indoor garage or showroom.
-   Image 1 was described as a portrait, while Image 2 was described as
    a product-style image.

These observations identify useful broad contrasts, although some scene
and purpose descriptions remain interpretive.

### 3.5 Object Comparison

The model stated that the only shared object was the red color and that
the dog and car were the only objects in their respective images.

This section contains a reasoning error: color is a visual attribute,
not an object. The response does not provide a reliable object-by-object
comparison.

### 3.6 Scene Comparison

The model generated the heading **"Scene Comparison"** but did not
provide any content under it. Therefore, no scene comparison was
completed in the returned output.

### 3.7 Overall Conclusion and Uncertainties

The returned output did not include the requested overall conclusion or
uncertainties sections. Their absence is an output-completeness
limitation.

## 4. Evaluation

  ------------------------------------------------------------------------
  Criterion               Assessment              Evidence from output
  ----------------------- ----------------------- ------------------------
  Individual image        Partially successful    Main subjects and broad
  descriptions                                    settings were described.

  Color identification    Successful in output    Red was identified in
                                                  both descriptions.

  Similarity reasoning    Partially successful    Shared color was
                                                  identified, but the
                                                  vehicle classification
                                                  was incorrect.

  Difference              Partially successful    Dog/car and
  identification                                  natural/indoor-setting
                                                  contrasts were
                                                  described.

  Object comparison       Weak                    Color was incorrectly
                                                  treated as an object.

  Scene comparison        Incomplete              Heading was present
                                                  without analysis.

  Overall conclusion      Not completed           Section was absent.

  Uncertainty reporting   Not completed           Section was absent.

  Instruction following   Incomplete              Not all requested
                                                  sections were returned.
  ------------------------------------------------------------------------

## 5. Limitations and Possible Causes

1.  **Object-category error:** The dog was incorrectly grouped with the
    car as a vehicle in the similarities section.
2.  **Attribute/object confusion:** The model treated the shared color
    as an object.
3.  **Unsupported interpretation:** Statements about a powerful engine,
    high performance, or readiness to move cannot necessarily be
    established from image appearance alone.
4.  **Scene uncertainty:** The model interpreted Image 2 as a garage or
    showroom, while also describing its background as blurred.
5.  **Incomplete response:** The output stopped after the Scene
    Comparison heading. The configured maximum of 450 new tokens may
    have contributed, but the available result does not establish the
    cause.

## 6. Recommendations for Improvement

-   Increase the maximum output-token limit and test whether all
    requested sections are completed.
-   Require the model to distinguish shared objects from shared
    attributes such as color, shape, and texture.
-   Ask for evidence supporting each similarity and difference.
-   Require uncertain interpretations to be explicitly labelled.
-   Validate that every required heading contains a substantive
    response.
-   Repeat the experiment to evaluate consistency.
-   Record GPU memory and utilization, if available, for a more complete
    performance profile.

## 7. Conclusion

Experiment 07 demonstrates that the Llama 3.2 11B Vision Instruct model
can generate basic descriptions of two images and identify some visible
contrasts. However, the result also contains an object-category error,
confusion between objects and visual attributes, potentially unsupported
interpretations, and incomplete sections.

**Overall assessment: Partially successful.** The experiment
demonstrates basic image-description capability, but cross-image
reasoning and output completeness need improvement.

## 8. Source

Source data: `experiment_07_image_comparison_results.json` supplied for
this report. The evaluation above is based on the recorded configuration
and model output. No independent human-verified image annotations or
ground-truth comparison were supplied.
