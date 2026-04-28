# POC Report Generation Guide

This guide outlines the procedure for generating a professional Proof of Concept (POC) report for the Ha-Meem AI Surveillance system. Generating a report involves two main parts: exporting actionable system data and compiling an executive presentation for stakeholders.

## 1. Generate the Data/Event Report

The system logs real-time events (such as tracking recognized or unknown faces) into the JSONL format. Stakeholders typically require a structured, readable ledger of these events.

- **Data Source:** `logs/events.jsonl`
- **Action:** Convert this raw data into a CSV or Excel format.
- **Data Points to Include:** 
  - Timestamp
  - Camera ID
  - Detected Identity (or "UNKNOWN")
  - Confidence Score
  - Path to Snapshot Evidence

*(Note: You can write a short Python script to parse `events.jsonl` and output a `poc_report.csv` automatically).*

## 2. Capture Performance Metrics

Stakeholders need to evaluate the technical efficiency of the system. While running your main pipeline (`py -m apps.multi_pipeline.main`), capture the following metrics:

- **Throughput:** Average FPS (Frames Per Second) maintained during inference.
- **Hardware Profile:** Document the specifications of the machine used (e.g., GPU model, CPU, RAM). This highlights the system's reliance on TensorRT/CUDA optimization.
- **Resource Utilization:** Record the CPU, GPU, and RAM usage percentages under maximum load.

## 3. Gather Visual Evidence

Visual proof is critical for demonstrating the system's capabilities.

- **Event Snapshots:** Select 3-4 clear examples from the `snapshots/` directory. Include examples of both successfully recognized individuals and captured unknown faces.
- **Live Interface:** Capture a screenshot or a short screen recording of the real-time multi-camera grid view to demonstrate the live tracking UI.

## 4. Compile the Executive Presentation

Create a slide deck or a PDF document structured as follows:

1. **Executive Summary:** A concise statement defining the system (a professional-grade AI surveillance system with real-time inference) and the success of the POC.
2. **System Architecture Overview:** Briefly map out the processing pipeline: 
   *Input Streams -> Face Detection -> Object Tracking -> Gallery Recognition -> Alert & Logging.*
3. **Performance Results:** Present the throughput (FPS), hardware specifications, and resource utilization gathered in Step 2.
4. **Visual Proof:** Showcase the live interface screenshots and the event snapshots gathered in Step 3.
5. **Current Capabilities:** Highlight key features such as multi-camera concurrency, TensorRT acceleration, and WhatsApp bot alerts (`apps.alert_bot.whatsapp_bot`).
6. **Limitations & Next Steps:** Transparently discuss any edge cases (e.g., severe occlusions, low light) and outline the requirements for moving to production (e.g., specific camera positioning, server procurement, or gallery expansion).
