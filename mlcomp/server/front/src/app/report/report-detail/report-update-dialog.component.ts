import {Component, Inject} from "@angular/core";
import {MAT_DIALOG_DATA, MatDialogRef} from "@angular/material";
import {ReportUpdateData} from "../../models";

@Component({
    selector: 'report-update-dialog',
    templateUrl: 'report-update-dialog.component.html',
})
export class ReportUpdateDialogComponent {

    constructor(
        public dialogRef: MatDialogRef<ReportUpdateDialogComponent>,
        @Inject(MAT_DIALOG_DATA) public data: ReportUpdateData
    ) {
        if(data.layouts.length>0){
            data.layout = data.layouts[0];
        }
    }

    on_no_click(): void {
        this.dialogRef.close();
    }

}