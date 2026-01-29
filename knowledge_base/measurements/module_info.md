---
category: vitals
type: measurement_reference
used_by: guidance_llm
version: 0.1
---

This document describes the physiological measurements available in TrackSanté,
their purpose, and when they are typically relevant based on user-reported symptoms.
This information is used by the guidance assistant to determine which measurements
may be appropriate to request.

---

## Heart Rate (HR)

**What it measures:**  
Beats per minute of the heart.

**Common relevance:**  
- Palpitations
- Chest discomfort
- Anxiety or stress
- Dizziness or fainting
- Fatigue
- Fever

**Important context to ask before measurement:**  
- Was the user resting or active?
- Recent caffeine, nicotine, or exercise
- Known heart conditions

---

## Blood Oxygen Saturation (SpO₂)

**What it measures:**  
Percentage of oxygen-saturated hemoglobin in the blood.

**Common relevance:**  
- Shortness of breath
- Chest tightness
- Respiratory symptoms
- Suspected infection
- Fatigue
- Dizziness

**Red flags:**  
- Very low oxygen levels
- Worsening shortness of breath at rest

---

## Body Temperature

**What it measures:**  
Core body temperature.

**Common relevance:**  
- Fever
- Chills
- Suspected infection
- Fatigue
- Inflammation

**Important context:**  
- Time of day
- Recent activity
- Use of fever-reducing medication

---

## Blood Pressure (BP)

**What it measures:**  
Systolic and diastolic arterial pressure.

**Common relevance:**  
- Headaches
- Dizziness
- Chest discomfort
- Known hypertension
- Vision changes

**Important context:**  
- Body position
- Recent activity
- Stress or anxiety
- Known blood pressure conditions

---

## General Notes for Guidance

- Not all vitals are required for every user.
- Measurements should only be suggested when relevant to the reported symptoms.
- The guidance assistant should collect sufficient contextual information before
  requesting measurements.
- This document does not define diagnostic thresholds or clinical interpretation.
