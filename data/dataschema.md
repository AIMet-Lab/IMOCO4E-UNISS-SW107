UNISS_SW_107 inference result payload schema
==============

## Global description
**Data schema describing the payload of messages produced and sent by SW-107 (AI for Sensor Anomaly Detection), developed by UNISS. Each message corresponds to a single inference result.**

## List of properties 

- `id`*(string)*: a unique text-based id
- `type`*(string)*: the type of inference result
- `anomalyDetected`*(boolean)*: indicates whether an anomaly has been detected
- `startTime`*(Date)*:  The timestamp of the start of the inference result. Following the ISO 8601 UTC format: YYYY-MM-DDThh:mm:ss.sssZ
- `endTime`*(Date)*:  The timestamp of the end of the inference result. Following the ISO 8601 UTC format: YYYY-MM-DDThh:mm:ss.sssZ
- `deviceId`*(string)*: a unique id of the device to which the inference result corresponds
- `MLModelId`*(string)*: a unique id of the ML model used to produce the inference result - versioning information is to be included in the id. 
- `threshold`*(double)*: a numerical threshold used to detect the anomaly
- `vre`*(double)*: the vector reconstruction error produced by the model.
- `timestamp`*(long)*:  The timestamp of each sensor data sample in **milliseconds since Unix epoch**. The timestamp should be the **current time** (i.e., the time when the sample is produced by the sensor simulator), but the **time difference** between individual consecutive samples in the original CSV file **should be respected**. The frequency appears to be quite consistent in the dataset at 250Hz and therefore, samples should be sent every 4ms.
- `pCut::Motor_Torque`*(double)*: Torque in nM. 
- `pCut::CTRL_Position_controller::Lag_error`*(double)*: Represent the instantaneous position error between the set-point from the path generator and the real current encoder position.
- `pCut::CTRL_Position_controller::Actual_position`*(int)*: Cutting blade position in mm. 
- `pCut::CTRL_Position_controller::Actual_speed`*(double)*: Speed of the cutting blade.
- `pSvolFilm::CTRL_Position_controller::Actual_position`*(int)*: Plastic film unwinder position in mm. 
- `pSvolFilm::CTRL_Position_controller::Actual_speed`*(double)*: Speed of the plastic film unwinder. 
- `pSvolFilm::CTRL_Position_controller::Lag_error`*(double)*: Represent the instantaneous position error between the set-point from the path generator and the real. 
- `pSpintor::VAX_speed`*(double)*: *Not present in one of the two datasets - No description provided*

## Example payload

```json  
{  
  "id": "ADE_2023-06-09T10:07:53.614Z",
  "type": "Anomaly Detection Event",
  "anomalyDetected": "true", 
  "startTime": "2023-06-09T10:07:53.614Z",
  "endTime": "2023-06-09T10:07:57.318Z",
  "deviceId": "P3_BladeSensor_004",
  "MLModelId": "CDAD_ReLUNode_[50-10-50]",
  "threshold": "0.035",
  "vre": "0.5",
  "timestamp": "1686733624571",
  "pCut::Motor_Torque": "0.28162419",
  "pCut::CTRL_Position_controller::Lag_error": "0.00250244",
  "pCut::CTRL_Position_controller::Actual_position": "628392625",
  "pCut::CTRL_Position_controller::Actual_speed": "-937.27111816",
  "pSvolFilm::CTRL_Position_controller::Actual_position": "5298565",
  "pSvolFilm::CTRL_Position_controller::Actual_speed": "2453.1909179",
  "pSvolFilm::CTRL_Position_controller::Lag_error": "0.87407165",
  "pSpintor::VAX_speed": "1379.99975585"
}

```