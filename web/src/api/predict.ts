import { useDiseaseStore } from "../store/diseaseStore"
import { useSymptomStore } from "../store/symptomStore"

export async function softmax_predict() : Promise<number[]> {
    const {model} = useDiseaseStore.getState()
    const {getHasSymptomList} = useSymptomStore.getState()
    const res = await fetch("http://localhost:5100/predict", {
        method : "POST",
        headers : {
            "content-type" : "application/json"
        },
        body: JSON.stringify(
            {
                symptomList : getHasSymptomList(),
                model
            }
        )
    })
    const data = await res.json()
    return data
}