import axios from 'axios';

const apiUrl = import.meta.env.VITE_API_URL;
const clientId = import.meta.env.VITE_CLIENT_ID;
const token = import.meta.env.VITE_CLIENT_TOKEN;

export const processFile = async (formData: FormData) => {
    return await axios.post(`${apiUrl}/llm-interaction-api/v1/process-file?client_id=${clientId}`, formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
            'Authorization': `${token}`
        },
    });
};