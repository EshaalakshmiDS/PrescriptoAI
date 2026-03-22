import axios from "axios";

const BASE = "http://localhost:8000/api";

export const uploadPrescription = (file) => {
  const form = new FormData();
  form.append("file", file);
  return axios.post(`${BASE}/upload-prescription`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

export const analyzeText = (text) =>
  axios.post(`${BASE}/analyze`, { text });

export const sendChat = (message, threadId, prescriptionContext) =>
  axios.post(`${BASE}/chat`, {
    message,
    thread_id: threadId,
    prescription_context: prescriptionContext,
  });
