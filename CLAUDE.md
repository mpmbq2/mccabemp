# CLAUDE.md - Project Overview

This project is a monorepo containing a Kedro data pipeline and a Quarto website.

## 1. Codebase (Kedro Project)

This directory contains a Kedro project for building data pipelines.

### Building and Running

*   **Install dependencies:**

    ```bash
    pip install -r codebase/requirements.txt
    ```

*   **Run the pipeline:**

    ```bash
    kedro run
    ```

*   **Test the project:**

    ```bash
    pytest
    ```

### Development Conventions

*   Follow data engineering conventions to ensure reproducibility.
*   Do not commit data or credentials to the repository.
*   Local configuration should be kept in `codebase/conf/local/`.

## 2. Website (Quarto Project)

This directory contains a Quarto website.

### Building and Running

*   **Preview the website:**

    ```bash
    quarto preview website
    ```

*   **Render the website:**

    ```bash
    quarto render website
    ```

### Key Files

*   `website/_quarto.yml`: Main configuration file for the Quarto project.
*   `website/index.qmd`: The homepage of the website.
*   `website/blog.qmd`: The blog page.
*   `website/posts/`: Directory containing the blog posts.
