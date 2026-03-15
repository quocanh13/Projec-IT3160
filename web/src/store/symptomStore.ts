import { create } from "zustand"
import symptomList from "../assets/data/symptom_data.json"
import type { Symptom } from "../types";

interface SymptomStore{
    symptomSearchTerm: string,
    symptomList: Symptom[],
    setSymptomSearchTerm: (symptomSearchTerm: string) => void,
    toggleSymptom: (id : number) => void,
    getHasSymptomList: () => number[],
    resetSymptom : () => void
}

export const useSymptomStore = create<SymptomStore>((set, get)=>({
    symptomSearchTerm: "",
    symptomList,

    setSymptomSearchTerm : (symptomSearchTerm)=>{
        set({symptomSearchTerm})
    },

    toggleSymptom(id: number){
        set(state => ({
            symptomList : state.symptomList.map(s => {
                if(s.id == id) return {...s, hasSymptom : !s.hasSymptom}
                else return s
            })
        }))
    },

    getHasSymptomList(){
        const {symptomList} = get()
        return symptomList.map(v => v.hasSymptom ? 1 : 0)
    },

    resetSymptom(){
        set({symptomList})
    }
}))