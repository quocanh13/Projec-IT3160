export async function softmax_predict(symptom: number[]) : Promise<number[]> {
    const res = await fetch("http://localhost:5000/softmax", {
        method : "POST",
        headers : {
            "content-type" : "application/json"
        },
        body: JSON.stringify(
            symptom
        )
    })
    const data = await res.json()
    return data
}