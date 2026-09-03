# Migration Guide: JITS to MTAP (Multimedia Tools and Applications)

This document outlines the detailed step-by-step instructions for converting the autonomous driving benchmarking article from the **Journal of Intelligent Transportation Systems (JITS)** format to the **Multimedia Tools and Applications (MTAP)** Springer Nature format.

---

## 1. Document Class and Packages (Preamble)

### JITS Setup
The JITS manuscript uses the Taylor & Francis `interact.cls` and APA-style citation packages:
```latex
\documentclass[suppldata]{interact}
\usepackage[natbibapa]{apacite}
\setlength\bibhang{12pt}
\renewcommand\bibliographytypesize{\fontsize{10}{12}\selectfont}
...
```

### MTAP Setup
MTAP requires the **Springer Nature LaTeX template (`sn-jnl.cls`)** with the numerical bibliography style option (**`sn-mathphys`** or similar):
```latex
\documentclass[sn-mathphys,pdflatex]{sn-jnl}

% Core Packages (already bundled or supported by sn-jnl)
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{siunitx}
\usepackage{booktabs}
\usepackage{float}
\usepackage{ragged2e}

% Avoid duplication of figures by pointing directly to the JITS folder
\graphicspath{ {../JITS/figures/} }
```

---

## 2. Title and Author Block Reformatting

JITS uses custom author commands (`\name`, `\affil`) with conditional double-blind anonymization. MTAP follows **single-blind peer review**, meaning the authors' details are included upon submission.

### JITS Formatting:
```latex
\title{Continuous Control for Vision-Based Lane Following: Benchmarking Deep Reinforcement Learning Algorithms}
\ifanonymous
\author{
\name{Anonymous Authors}
\affil{Affiliations omitted for double-blind peer review}
}
\else
\author{
\name{Rub{\'e}n Lucas Zaragoza\textsuperscript{a}\thanks{Corresponding author. Email: r.lucasz@alumnos.urjc.es} \and Jos{\'e} Mar{\'\i}a Ca{\~n}as Plaza\textsuperscript{a}}
\affil{\textsuperscript{a}Universidad Rey Juan Carlos, Madrid, Spain}
}
\fi
```

### MTAP Formatting (`sn-jnl` syntax):
```latex
\title[Vision-Based Lane Following Continuous Control]{Continuous Control for Vision-Based Lane Following: Benchmarking Deep Reinforcement Learning Algorithms}

\author*[1]{\fnm{Rub\'en} \sur{Lucas Zaragoza}}\email{r.lucasz@alumnos.urjc.es}
\author[1]{\fnm{Jos\'e Mar\'ia} \sur{Ca\~nas Plaza}}\email{josemaria.canas@urjc.es}

\affil[1]{\orgdiv{Departamento de Tecnolog\'ia Electr\'onica}, \orgname{Universidad Rey Juan Carlos}, \orgaddress{\city{M\'ostoles}, \postcode{28933}, \state{Madrid}, \country{Spain}}}
```

---

## 3. Abstract and Keywords

The abstract and keywords block needs to be wrapped inside `sn-jnl`'s abstract environment.
- **Abstract word count limit:** 150 to 250 words (current is ~230 words, which is perfect).
- **Keywords count limit:** 4 to 6 keywords (current has 5, which is perfect).

```latex
\begin{abstract}
Autonomous vehicles require robust decision-making and continuous control systems...
\end{abstract}

\keywords{Autonomous Driving, Deep Reinforcement Learning, Lane Following, Algorithm Benchmarking, YOLOP}
```

---

## 4. Citation and References Conversion

This is the most critical manual/semi-automated task.
- JITS uses **APA style** (`apacite`) with author-year citations: `\citep{...}` and `\citet{...}`.
- MTAP uses **Numerical style** (`[1]`, `[1-3]`) with standard `\cite{...}`.

### Conversion Mapping:
1. Replace all parenthetical citations:
   - **From:** `\citep{key}` (renders as `(Author, Year)`)
   - **To:** `\cite{key}` (renders as `[1]`)
2. Replace textual citations:
   - **From:** `\citet{key}` (renders as `Author (Year)`)
   - **To:** `Author et al. \cite{key}` or similar contextual narrative + `\cite{key}`
3. Remove JITS-specific APA configurations from the preamble.
4. Replace JITS bibliography commands at the end of the file:
   - **From:**
     ```latex
     \bibliographystyle{tfp}
     \bibliography{references}
     ```
   - **To:** (using JITS bib files to avoid duplication)
     ```latex
     \bibliography{../JITS/references}
     ```

---

## 5. Headings Structure

MTAP guidelines state: *"Please use the decimal system of headings with no more than three levels."*
- Level 1: `\section{...}` (renders as **1**, **2**)
- Level 2: `\subsection{...}` (renders as **1.1**, **1.2**)
- Level 3: `\subsubsection{...}` (renders as **1.1.1**)
- Do not use paragraph-level subheadings with custom symbols.

---

## 6. Mandatory "Declarations" Section

Springer Nature strictly requires a **Declarations** section before the reference list.

```latex
\section*{Declarations}

\subsection*{Funding}
The authors did not receive support from any organization for the submitted work. \textit{[Or provide actual grant info if applicable]}

\subsection*{Competing Interests}
The authors have no relevant financial or non-financial interests to disclose.

\subsection*{Ethics Approval}
Not applicable.

\subsection*{Consent to Participate}
Not applicable.

\subsection*{Consent for Publication}
Not applicable.

\subsection*{Availability of Data and Materials}
The datasets generated and/or analyzed during the current study are available from the corresponding author on reasonable request.

\subsection*{Code Availability}
The code and configuration files used in this study are available in the open-source repository RL-Studio at \url{https://github.com/JdeRobot/RL-Studio}.

\subsection*{Authors' Contributions}
All authors contributed to the study conception and design. Material preparation, data collection, and analysis were performed by Rub\'en Lucas Zaragoza. The first draft of the manuscript was written by Rub\'en Lucas Zaragoza, and Jos\'e Mar\'ia Ca\~nas Plaza supervised and critically revised the work. All authors read and approved the final manuscript.
```

---

## 7. Operational Workflow for Execution

When we begin migration:
1. **Download the Official Template:**
   Download and unpack the latest `sn-jnl.cls` directly into the `MTAP/` folder.
2. **Initialize Manuscript File:**
   Create `MTAP/MTAP_manuscript.tex` from the raw text of `JITS/JITS_7_0726.tex`.
3. **Execute Citations RegEx/Replacement:**
   Search and replace `\citep` with `\cite`, and adjust `\citet` citations into readable inline text + `\cite`.
4. **Draft the Declarations Section:**
   Insert the required Declarations block right before `\bibliography{...}`.
5. **Verify and Compile:**
   Compile the document locally to verify that:
   - Figures are resolved correctly from `../JITS/figures/`.
   - References are resolved correctly from `../JITS/references.bib`.
   - There are no class-specific compile issues.
