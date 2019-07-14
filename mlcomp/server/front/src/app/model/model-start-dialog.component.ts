import {Component, Inject} from "@angular/core";
import {MAT_DIALOG_DATA, MatDialogRef} from "@angular/material";
import {ModelStartData} from "../models";

@Component({
    selector: 'model-start-dialog',
    templateUrl: 'model-start-dialog.html',
})
export class ModelStartDialogComponent {

    constructor(
        public dialogRef: MatDialogRef<ModelStartDialogComponent>,
        @Inject(MAT_DIALOG_DATA) public data: ModelStartData) {
        this.dag_changed();
    }

    onNoClick(): void {
        this.dialogRef.close();
    }

    dag_changed() {
        let dag = this.data.dag;
        if (!dag) {
            return;
        }

        if (dag.slots.length == 1) {
            this.data.slot = dag.slots[0];
        }
        if (dag.interfaces.length == 1) {
            this.data.interface = dag.interfaces[0];
        }
        if (dag.pipes.length == 1) {
            this.data.pipe = dag.pipes[0];
        }
    }
}