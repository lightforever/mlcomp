import {Component, Inject} from "@angular/core";
import {MAT_DIALOG_DATA, MatDialogRef} from "@angular/material";
import {ModelStartData} from "../models";
import {ModelService} from "./model.service";

@Component({
    selector: 'model-start-dialog',
    templateUrl: 'model-start-dialog.html',
})
export class ModelStartDialogComponent {
    error: string;

    constructor(
        public dialogRef: MatDialogRef<ModelStartDialogComponent>,
        @Inject(MAT_DIALOG_DATA) public data: ModelStartData,
        protected service: ModelService
    ) {
        this.dag_changed();
    }

    on_ok_click(): void{
        this.service.start(this.data).subscribe(res=>{
            this.error = res.error;
            if(res.success){
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

        if (dag.slots.length >= 1) {
            this.data.slot = dag.slots[0];
        }
        if (dag.interfaces.length >= 1) {
            this.data.interface = dag.interfaces[0];
        }
        if (dag.pipes.length >= 1) {
            this.data.pipe = dag.pipes[0];
        }
    }
}