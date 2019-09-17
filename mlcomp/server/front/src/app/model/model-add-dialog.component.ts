import {Component, Inject} from "@angular/core";
import {MAT_DIALOG_DATA, MatDialogRef} from "@angular/material";
import {ModelAddData} from "../models";
import {ModelService} from "./model.service";

@Component({
    selector: 'model-add-dialog',
    templateUrl: 'model-add-dialog.html',
})
export class ModelAddDialogComponent {
    error: string;

    constructor(
        public dialogRef: MatDialogRef<ModelAddDialogComponent>,
        @Inject(MAT_DIALOG_DATA) public data: ModelAddData,
        public service: ModelService) {
    }

    on_ok_click(): void{
        this.service.add(this.data).subscribe(res=>{
            this.error = res.error;
            if(res.success){
                this.dialogRef.close();
            }
        });
    }

    on_cancel_click(): void {
        this.dialogRef.close();
    }
}