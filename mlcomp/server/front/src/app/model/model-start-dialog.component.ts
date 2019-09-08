import {Component, Inject, OnInit} from "@angular/core";
import {MAT_DIALOG_DATA, MatDialogRef} from "@angular/material";
import {ModelStartData} from "../models";
import {ModelService} from "./model.service";

@Component({
    selector: 'model-start-dialog',
    templateUrl: 'model-start-dialog.html',
})
export class ModelStartDialogComponent implements OnInit {
    error: string;

    constructor(
        public dialogRef: MatDialogRef<ModelStartDialogComponent>,
        @Inject(MAT_DIALOG_DATA) public data: ModelStartData,
        protected service: ModelService
    ) {

    }

    on_ok_click(): void {
        this.service.start_end(this.data).subscribe(res => {
            this.error = res.error;
            if (res.success) {
                this.dialogRef.close();
            }
        });
    }

    on_cancel_click(): void {
        this.dialogRef.close();
    }

    dag_changed() {
        let dag = this.data.dag;
        if (!dag) {
            return;
        }
        if (dag.pipes.length >= 1) {
            this.data.pipe = dag.pipes[0];
        }
    }

    ngOnInit(): void {
        let self = this;
        this.service.start_begin(this.data).subscribe(res => {
            this.error = res.error;
            if(this.error){
                return;
            }
            this.data = {
                'model_id': self.data.model_id,
                'dags': res.dags
            } as ModelStartData;

            if(res.dag){
                for(let d of this.data.dags){
                    if(d.id == res.dag.id){
                        this.data.dag = d;
                        break
                    }
                }
            }

            this.dag_changed();
        })
    }
}