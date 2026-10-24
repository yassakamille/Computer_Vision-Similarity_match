# Computer Vision 
### Similarity match
[![Architecture diagram of yassakamille/computer_vision-similarity_match](https://gitdiagram.com/yassakamille/computer_vision-similarity_match/diagram.png)](https://gitdiagram.com/yassakamille/computer_vision-similarity_match?utm_source=readme&utm_medium=picture)

## Image Similarity and Feature Matching using OpenCV

This project demonstrates image similarity analysis using **SIFT feature extraction, descriptor matching, ratio filtering, and RANSAC-based homography estimation**. It visualizes feature correspondences and geometrically consistent matches, then calculates a similarity score to classify two images.

### Workflow Diagram

```mermaid
flowchart TD
    subgraph matching["Feature Matching"]
        similarity["Similarity Example<br/>SIFT/Code.py"]
        sift["SIFT Feature Extraction"]
        ratio["Ratio-Filtered Matches"]
        ransac["RANSAC Matching Demo<br/>RANSAC/matching_RANSAC.py"]
        basic["Basic Matching Demo<br/>RANSAC/matching.py"]
    end

    subgraph geometry["Geometric Verification"]
        homography["RANSAC Homography Estimation"]
    end

    subgraph results["Results and Visualization"]
        match_display["Match Visualization"]
        inlier_display["Inlier Visualization"]
        score["Similarity Score and Decision"]
        similarity_display["Inlier Match Display"]
    end

    images["Input Images"]

    images --> similarity
    similarity --> sift
    sift --> ratio
    ratio --> homography
    ratio --> score
    homography --> similarity_display

    images --> ransac
    ransac --> homography
    ransac --> inlier_display

    images --> basic
    basic --> match_display

    click similarity "https://github.com/yassakamille/computer_vision-similarity_match/blob/main/SIFT/Code.py"
    click ransac "https://github.com/yassakamille/computer_vision-similarity_match/blob/main/RANSAC/matching_RANSAC.py"
    click basic "https://github.com/yassakamille/computer_vision-similarity_match/blob/main/RANSAC/matching.py"

    classDef process fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#172554
    classDef geometryStyle fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#78350f
    classDef result fill:#dcfce7,stroke:#16a34a,stroke-width:1.5px,color:#14532d
    classDef inputStyle fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,color:#312e81

    class similarity,sift,ratio,ransac,basic process
    class homography geometryStyle
    class match_display,inlier_display,score,similarity_display result
    class images inputStyle
```

### Main Techniques

- **SIFT:** Detects distinctive keypoints and computes feature descriptors.
- **Brute-Force Matching:** Matches descriptors between the two images.
- **Ratio Test:** Filters ambiguous matches using the nearest and second-nearest descriptor distances.
- **RANSAC Homography:** Estimates a geometric transformation while rejecting inconsistent correspondences.
- **Match Visualization:** Displays corresponding keypoints between images.
- **Similarity Classification:** Calculates a similarity score and compares it against a predefined threshold.

### Implementation Note

The provided code estimates a homography but does not apply it to create a warped image using `cv2.warpPerspective()`. The diagram therefore focuses on the implemented matching, verification, and visualization workflow.

The similarity score is a heuristic based on the number of ratio-filtered matches relative to the smaller keypoint count; it is not a universal measure of image similarity.
[![Architecture diagram](https://gitdiagram.com/diagram-badge.svg)](https://gitdiagram.com/yassakamille/computer_vision-similarity_match?utm_source=readme&utm_medium=badge)
