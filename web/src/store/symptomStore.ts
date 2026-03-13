import { create } from "zustand"
interface SymptomStore{
    symptomSearchTerm: string,
    setSymptomSearchTerm: (symptomSearchTerm: string) => void
}

export const useSymptomStore = create<SymptomStore>((set)=>({
    symptomSearchTerm: "",
    setSymptomSearchTerm : (symptomSearchTerm)=>{
        set({symptomSearchTerm})
    }
}))