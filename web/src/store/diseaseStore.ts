import { create } from "zustand"
import diseaseList from "../assets/data/disease_data.json" 
import type { Disease } from "../types"

interface DiseaseStore{
    diseaseList : Disease[],
    model : string,
    setRiskPercent: (riskList: number[]) => void,
    setModel : (model : string) => void
}

export const useDiseaseStore = create<DiseaseStore>((set) => ({
    diseaseList,
    model : "softmax logistic",
    setRiskPercent(riskList) {
        set(state => ({
            diseaseList : state.diseaseList.map((v, i) => ({...v, riskPercent : riskList[i]}))   
        }))
    },
    setModel(model) {
        set({model})
    },
}))