import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface AuthState {
    token: string | null;
}

const initialState: AuthState = {
    token: localStorage.getItem('token'),
};

const authSlice = createSlice({
    name: 'auth',
    initialState,
    reducers: {
        login: (state: any, action: PayloadAction<string>) => {
            state.token = action.payload;
            localStorage.setItem('token', action.payload);
        },
        logout: (state: any) => {
            state.token = null;
            localStorage.removeItem('token');
        },
    },
});

export const { login, logout } = authSlice.actions;
export default authSlice.reducer;