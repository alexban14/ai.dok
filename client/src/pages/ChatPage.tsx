import React, { useState } from "react";
import axios from "axios";
import {
    Card,
    CardContent,
    Typography,
    TextField,
    Button,
    CircularProgress,
    Box
} from "@mui/material";
import { processFile } from "../api/api.ts";

const ChatPage: React.FC = () => {
    const [prompt, setPrompt] = useState("");
    const [file, setFile] = useState<File | null>(null);
    const [response, setResponse] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const apiUrl = import.meta.env.VITE_API_URL;

    const handleSubmit = async () => {
        if (!file || !prompt) return;

        setLoading(true);
        setResponse(null);

        const formData = new FormData();
        formData.append("ai_service", "groq_cloud");
        formData.append("model", "llama-3.3-70b-versatile");
        formData.append("processing_type", "prompt");
        formData.append("prompt", prompt);
        formData.append("file", file);

        try {
            const res = await processFile(formData);

            setResponse(res.data.response);
        } catch (error) {
            console.error("Error:", error);
            setResponse("An error occurred. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <Box
            display="flex"
            flexDirection="column"
            alignItems="center"
            justifyContent="center"
            width="100vw"
            height="100vh"
            bgcolor="#f5f5f5"
            p={3}
        >
            <Card
                sx={{
                    width: "100%",
                    maxWidth: "800px",
                    p: 4,
                    boxShadow: 3,
                    bgcolor: "white",
					overflow: "auto"
                }}
            >
                <CardContent>
                    <Typography variant="h5" textAlign="center" fontWeight="bold" mb={3}>
						AI-DOK Medical Assistant
                    </Typography>

                    <TextField
                        label="Enter your prompt..."
                        multiline
                        rows={3}
                        fullWidth
                        variant="outlined"
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        sx={{ mb: 3 }}
                    />

                    <Button
                        variant="contained"
                        component="label"
                        fullWidth
                        sx={{ mb: 2 }}
                    >
                        Upload File
                        <input
                            type="file"
                            hidden
                            accept=".txt,.pdf,.docx"
                            onChange={(e) => setFile(e.target.files?.[0] || null)}
                        />
                    </Button>

                    {file && (
                        <Typography variant="body2" color="text.secondary" textAlign="center" mb={2}>
                            Selected file: {file.name}
                        </Typography>
                    )}

                    <Button
                        variant="contained"
                        color="primary"
                        fullWidth
                        onClick={handleSubmit}
                        disabled={loading || !file || !prompt}
                        sx={{ py: 1.5 }}
                    >
                        {loading ? <CircularProgress size={24} sx={{ color: "#fff" }} /> : "Submit"}
                    </Button>

                    {response && (
                        <Box
                            mt={3}
                            p={2}
                            bgcolor="#e3f2fd"
                            borderRadius={2}
                            borderLeft={4}
                            borderColor="primary.main"
							overflow="auto"
                        >
                            <Typography variant="body1" dangerouslySetInnerHTML={{ __html: response }} />
                        </Box>
                    )}
                </CardContent>
            </Card>
        </Box>
    );
};

export default ChatPage;