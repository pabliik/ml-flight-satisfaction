# Dataset Description — Airline Passenger Satisfaction

## Overview

| Property | Value |
|----------|-------|
| Samples | 129,879 |
| Features | 24 |
| Task | Binary Classification |
| Target | `Satisfaction` (Satisfied / Neutral or Unsatisfied) |

---

## Features

### Passenger Information

| Field | Type | Description |
|-------|------|-------------|
| `ID` | Numerical | Unique passenger identifier |
| `Gender` | Categorical | Female / Male |
| `Age` | Numerical | Age of the passenger |
| `Customer Type` | Categorical | First-time / Returning |
| `Type of Travel` | Categorical | Business / Personal |
| `Class` | Categorical | Travel class (Business / Eco / Eco Plus) |

### Flight Information

| Field | Type | Description |
|-------|------|-------------|
| `Flight Distance` | Numerical | Flight distance in miles |
| `Departure Delay` | Numerical | Departure delay in minutes |
| `Arrival Delay` | Numerical | Arrival delay in minutes |

### Satisfaction Ratings (1–5 scale)

> **Note:** All rating features use a scale from **1** (lowest) to **5** (highest). A value of **0** means "not applicable".

| Field | Description |
|-------|-------------|
| `Departure and Arrival Time Convenience` | Satisfaction with flight departure and arrival times |
| `Ease of Online Booking` | Satisfaction with the online booking experience |
| `Check-in Service` | Satisfaction with the check-in service |
| `Online Boarding` | Satisfaction with the online boarding experience |
| `Gate Location` | Satisfaction with the gate location in the airport |
| `On-board Service` | Satisfaction with the on-board service |
| `Seat Comfort` | Satisfaction with the comfort of the airplane seat |
| `Leg Room Service` | Satisfaction with the leg room of the seat |
| `Cleanliness` | Satisfaction with the cleanliness of the airplane |
| `Food and Drink` | Satisfaction with food and drinks on the airplane |
| `In-flight Service` | Satisfaction with the in-flight service |
| `In-flight Wifi Service` | Satisfaction with the in-flight Wifi service |
| `In-flight Entertainment` | Satisfaction with the in-flight entertainment |
| `Baggage Handling` | Satisfaction with baggage handling from the airline |

### Target Variable

| Field | Type | Description |
|-------|------|-------------|
| `Satisfaction` | Categorical | Overall satisfaction — **Satisfied** or **Neutral or Unsatisfied** |
