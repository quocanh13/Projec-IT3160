import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import {DiseasePrediction} from './DiseasePrediction.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <DiseasePrediction />
  </StrictMode>,
)
